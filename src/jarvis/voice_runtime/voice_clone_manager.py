from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

from jarvis.config import Settings

from .audio_preprocessor import AudioPreprocessor
from .quality import score_sample_quality
from .sample_validator import VoiceSampleValidator
from .voice_profile import VoiceProfile, VoiceSampleMetadata, build_default_voice_profiles, resolve_profile_name


class VoiceCloneManager:
    def __init__(self, settings: Settings, *, logger: logging.Logger | None = None) -> None:
        self._settings = settings
        self._logger = logger or logging.getLogger("jarvis.voice.clone")
        self._validator = VoiceSampleValidator()
        self._preprocessor = AudioPreprocessor()
        self._cache_lock = Lock()
        self._profile_cache: dict[tuple[object, ...], VoiceProfile] = {}

    def resolve_profile(self, profile_name: str | None = None) -> VoiceProfile:
        profiles = build_default_voice_profiles(self._settings)
        selected_name = resolve_profile_name(self._settings, profile_name)
        profile = profiles.get(selected_name, profiles["jarvis_premium"])
        cache_key = self._profile_cache_key(profile, profile_name)
        with self._cache_lock:
            cached = self._profile_cache.get(cache_key)
        if cached is not None:
            return cached
        self._logger.info(
            "profile_selected",
            extra={
                "profile_requested": profile_name or selected_name,
                "profile_selected": profile.name,
                "clone_backend": profile.backend,
                "fallback_order": list(profile.fallback_backends),
            },
        )
        if not profile.clone_enabled or profile.speaker_wav is None:
            if profile.name == "fallback_basic":
                self._store_profile_cache(cache_key, profile)
                return profile
            warnings = tuple(dict.fromkeys([*profile.warnings, "voice_clone_disabled_or_missing_sample"]))
            resolved = replace(profile, clone_ready=False, validation_status="disabled", warnings=warnings)
            self._store_profile_cache(cache_key, resolved)
            return resolved
        validation = self._validator.validate(profile.speaker_wav)
        self._logger.info(
            "clone_sample_validated",
            extra={
                "profile_name": profile.name,
                "sample_path": str(profile.speaker_wav),
                "valid": validation.valid,
                "status": validation.status,
                "quality_message": validation.message,
                "warnings": list(validation.warnings),
            },
        )
        prepared_path = validation.metadata.path
        prepared_validation = validation
        if validation.valid and self._settings.voice_clone_preprocess_enabled and validation.metadata.path is not None:
            try:
                output_dir = self._settings.resolved_data_dir / "voice_clone"
                prepared_path, prepared_validation = self._preprocessor.prepare(validation.metadata.path, output_dir=output_dir)
                self._logger.info(
                    "clone_preprocess_output",
                    extra={
                        "profile_name": profile.name,
                        "input_path": str(validation.metadata.path),
                        "output_path": str(prepared_path),
                        "valid": prepared_validation.valid,
                        "status": prepared_validation.status,
                        "quality_message": prepared_validation.message,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("voice_clone_preprocess_failed", extra={"error": str(exc), "sample_path": str(validation.metadata.path)})
        quality_score = score_sample_quality(prepared_validation if self._settings.voice_clone_quality_check_enabled else validation)
        self._logger.info(
            "clone_sample_quality_score",
            extra={
                "profile_name": profile.name,
                "quality_score": quality_score,
                "threshold": self._settings.voice_clone_quality_threshold,
                "quality_check_enabled": self._settings.voice_clone_quality_check_enabled,
            },
        )
        clone_ready = prepared_validation.valid
        validation_status = "ready" if clone_ready else prepared_validation.status
        metadata = _merge_metadata(validation.metadata, prepared_validation.metadata, prepared_path)
        warnings = tuple(dict.fromkeys([*profile.warnings, *validation.warnings, *prepared_validation.warnings]))
        if self._settings.voice_clone_quality_check_enabled and quality_score < self._settings.voice_clone_quality_threshold:
            warnings = tuple(dict.fromkeys([*warnings, "quality_below_threshold"]))
            clone_ready = False
            validation_status = "quality_rejected"
        resolved = replace(
            profile,
            speaker_wav=prepared_path if clone_ready else profile.speaker_wav,
            quality_score=quality_score,
            validation_status=validation_status,
            sample_message=prepared_validation.message,
            sample_metadata=metadata,
            sample_recommendations=prepared_validation.recommendations or validation.recommendations,
            clone_ready=clone_ready,
            warnings=warnings,
        )
        self._store_profile_cache(cache_key, resolved)
        return resolved

    def _profile_cache_key(self, profile: VoiceProfile, requested_name: str | None) -> tuple[object, ...]:
        sample = profile.speaker_wav
        sample_path = str(sample or "")
        sample_mtime: int | None = None
        if sample is not None:
            try:
                if sample.exists():
                    sample_path = str(sample.resolve())
                    sample_mtime = sample.stat().st_mtime_ns
            except OSError:
                sample_mtime = None
        return (
            requested_name or profile.name,
            profile.name,
            profile.backend,
            profile.clone_enabled,
            sample_path,
            sample_mtime,
            self._settings.voice_clone_preprocess_enabled,
            self._settings.voice_clone_quality_check_enabled,
            self._settings.voice_clone_quality_threshold,
        )

    def _store_profile_cache(self, cache_key: tuple[object, ...], profile: VoiceProfile) -> None:
        with self._cache_lock:
            self._profile_cache[cache_key] = profile

    def build_request_metadata(self, profile: VoiceProfile) -> dict[str, object]:
        return {
            "voice_profile": profile.name,
            "voice_clone_enabled": profile.clone_enabled,
            "voice_clone_ready": profile.clone_ready,
            "voice_clone_backend": profile.backend,
            "voice_clone_quality_score": profile.quality_score,
            "voice_clone_validation_status": profile.validation_status,
            "voice_clone_warnings": list(profile.warnings),
            "voice_clone_message": profile.sample_message,
            "voice_clone_recommendations": list(profile.sample_recommendations),
            "speaker_name": profile.speaker_name,
            "speaker_wav": str(profile.speaker_wav) if profile.clone_ready and profile.speaker_wav is not None else None,
            "speaker_wav_effective": str(profile.speaker_wav) if profile.clone_ready and profile.speaker_wav is not None else None,
            "speaking_rate": profile.speaking_rate,
            "style": profile.style,
            "pause_ms": profile.pause_ms,
            "short_pause_ms": profile.short_pause_ms,
            "medium_pause_ms": profile.medium_pause_ms,
            "final_pause_ms": profile.final_pause_ms,
            "pause_style": profile.pause_style,
            "formality_level": profile.formality_level,
            "language_style": profile.language_style,
            "tone": profile.tone,
            "tts_provider_candidates": [profile.backend, *profile.fallback_backends],
        }

    def status(self, profile_name: str | None = None, *, resolve: bool = True) -> dict[str, Any]:
        if not resolve:
            return self.lightweight_status(profile_name)
        profile = self.resolve_profile(profile_name)
        return {
            "enabled": self._settings.voice_clone_enabled,
            "active_profile": profile.name,
            "backend": profile.backend,
            "clone_ready": profile.clone_ready,
            "validation_status": profile.validation_status,
            "quality_score": profile.quality_score,
            "speaker_wav": str(profile.speaker_wav) if profile.speaker_wav is not None else None,
            "speaker_wav_effective": str(profile.speaker_wav) if profile.clone_ready and profile.speaker_wav is not None else None,
            "sample_message": profile.sample_message,
            "sample_recommendations": list(profile.sample_recommendations),
            "warnings": list(profile.warnings),
            "sample_metadata": _metadata_to_dict(profile.sample_metadata),
        }

    def lightweight_status(self, profile_name: str | None = None) -> dict[str, Any]:
        profiles = build_default_voice_profiles(self._settings)
        selected_name = resolve_profile_name(self._settings, profile_name)
        profile = profiles.get(selected_name, profiles["jarvis_premium"])
        return {
            "enabled": self._settings.voice_clone_enabled,
            "active_profile": profile.name,
            "backend": profile.backend,
            "clone_ready": False,
            "validation_status": "not_loaded",
            "quality_score": None,
            "speaker_wav": str(profile.speaker_wav) if profile.speaker_wav is not None else None,
            "speaker_wav_effective": None,
            "sample_message": "",
            "sample_recommendations": [],
            "warnings": list(profile.warnings),
            "sample_metadata": _metadata_to_dict(VoiceSampleMetadata(path=profile.speaker_wav, exists=bool(profile.speaker_wav and profile.speaker_wav.exists()))),
        }


def _merge_metadata(base: VoiceSampleMetadata, prepared: VoiceSampleMetadata, prepared_path: Path | None) -> VoiceSampleMetadata:
    return VoiceSampleMetadata(
        path=prepared_path or prepared.path or base.path,
        original_path=base.original_path or base.path,
        exists=prepared.exists or base.exists,
        container=prepared.container or base.container,
        sample_rate=prepared.sample_rate or base.sample_rate,
        channels=prepared.channels or base.channels,
        duration_seconds=prepared.duration_seconds or base.duration_seconds,
        peak_level=prepared.peak_level if prepared.peak_level is not None else base.peak_level,
        rms_level=prepared.rms_level if prepared.rms_level is not None else base.rms_level,
        silence_ratio=prepared.silence_ratio if prepared.silence_ratio is not None else base.silence_ratio,
        clipped_ratio=prepared.clipped_ratio if prepared.clipped_ratio is not None else base.clipped_ratio,
        noise_floor_db=prepared.noise_floor_db if prepared.noise_floor_db is not None else base.noise_floor_db,
        warnings=tuple(dict.fromkeys([*base.warnings, *prepared.warnings])),
    )


def _metadata_to_dict(metadata: VoiceSampleMetadata) -> dict[str, Any]:
    return {
        "path": str(metadata.path) if metadata.path is not None else None,
        "original_path": str(metadata.original_path) if metadata.original_path is not None else None,
        "exists": metadata.exists,
        "container": metadata.container,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "duration_seconds": metadata.duration_seconds,
        "peak_level": metadata.peak_level,
        "rms_level": metadata.rms_level,
        "silence_ratio": metadata.silence_ratio,
        "clipped_ratio": metadata.clipped_ratio,
        "noise_floor_db": metadata.noise_floor_db,
        "warnings": list(metadata.warnings),
    }
