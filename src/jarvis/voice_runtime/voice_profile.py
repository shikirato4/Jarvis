from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from jarvis.config import Settings


@dataclass(slots=True, frozen=True)
class VoiceSampleMetadata:
    path: Path | None = None
    original_path: Path | None = None
    exists: bool = False
    container: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    peak_level: float | None = None
    rms_level: float | None = None
    silence_ratio: float | None = None
    clipped_ratio: float | None = None
    noise_floor_db: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class VoiceSampleValidation:
    valid: bool
    status: str
    message: str
    metadata: VoiceSampleMetadata = field(default_factory=VoiceSampleMetadata)
    warnings: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class VoiceProfile:
    name: str
    backend: str
    fallback_backends: tuple[str, ...]
    language_style: str
    style: str
    rate: int
    tone: str
    cleanup_enabled: bool
    style_enabled: bool
    formality_level: int
    language: str = "es"
    speaker_name: str | None = None
    speaker_wav: Path | None = None
    speaking_rate: float | None = None
    pause_ms: int = 180
    short_pause_ms: int = 120
    medium_pause_ms: int = 220
    final_pause_ms: int = 320
    pause_style: str = "balanced"
    quality_score: float | None = None
    validation_status: str = "unvalidated"
    sample_message: str = ""
    sample_metadata: VoiceSampleMetadata = field(default_factory=VoiceSampleMetadata)
    sample_recommendations: tuple[str, ...] = ()
    clone_enabled: bool = False
    clone_ready: bool = False
    warnings: tuple[str, ...] = ()


def build_default_voice_profiles(settings: Settings) -> dict[str, VoiceProfile]:
    clone_sample = settings.resolved_voice_clone_sample_path or settings.resolved_voice_coqui_speaker_wav
    speaker_name = settings.voice_coqui_speaker_name or settings.voice_default_voice_name
    clone_backend = settings.voice_clone_backend_default or settings.voice_tts_provider_default
    fallback_backends = _profile_fallbacks(settings, clone_backend)
    stable_rate = min(max(settings.voice_tts_rate, 158), 172)
    speaking_rate = min(max(settings.voice_speaking_rate, 0.82), 1.05)
    short_pause_ms, medium_pause_ms, final_pause_ms = _pause_values(settings.voice_pause_style)
    premium = VoiceProfile(
        name="jarvis_premium",
        backend=clone_backend,
        fallback_backends=fallback_backends,
        language_style="technical_formal",
        style="serious_precise_cinematic",
        rate=stable_rate,
        tone="formal_cinematic",
        cleanup_enabled=settings.voice_cleanup_enabled,
        style_enabled=settings.voice_style_enabled,
        formality_level=max(settings.voice_formality_level, 5),
        language=settings.voice_default_language,
        speaker_name=speaker_name,
        speaker_wav=clone_sample,
        speaking_rate=min(max(speaking_rate, 0.88), 0.92),
        pause_ms=medium_pause_ms,
        short_pause_ms=short_pause_ms,
        medium_pause_ms=medium_pause_ms,
        final_pause_ms=final_pause_ms,
        pause_style=settings.voice_pause_style,
        clone_enabled=settings.voice_clone_enabled,
    )
    cinematic = replace(
        premium,
        name="jarvis_cinematic",
        language_style="technical_formal_cinematic",
        style="cinematic_authoritative",
        tone="cinematic_serious",
        speaking_rate=min(max(speaking_rate, 0.88), 0.90),
        pause_ms=max(medium_pause_ms, 240),
    )
    custom = VoiceProfile(
        name="custom_clone",
        backend=clone_backend,
        fallback_backends=fallback_backends,
        language_style="custom_clone_formal",
        style="custom_clone",
        rate=settings.voice_tts_rate,
        tone="custom",
        cleanup_enabled=settings.voice_cleanup_enabled,
        style_enabled=settings.voice_style_enabled,
        formality_level=max(settings.voice_formality_level, 3),
        language=settings.voice_default_language,
        speaker_name=speaker_name,
        speaker_wav=clone_sample,
        speaking_rate=max(speaking_rate, 0.95),
        pause_ms=180,
        short_pause_ms=max(short_pause_ms - 20, 80),
        medium_pause_ms=max(medium_pause_ms - 30, 120),
        final_pause_ms=max(final_pause_ms - 40, 180),
        pause_style=settings.voice_pause_style,
        clone_enabled=settings.voice_clone_enabled,
    )
    fallback = VoiceProfile(
        name="fallback_basic",
        backend="pyttsx3" if "pyttsx3" in settings.voice_tts_provider_fallback_order or settings.voice_tts_provider_default == "pyttsx3" else "in_memory",
        fallback_backends=("in_memory",),
        language_style="fallback_clear",
        style="fallback_clear",
        rate=min(settings.voice_tts_rate + 12, 220),
        tone="clear",
        cleanup_enabled=True,
        style_enabled=False,
        formality_level=max(settings.voice_formality_level - 1, 2),
        language=settings.voice_default_language,
        speaker_name=settings.voice_default_voice_name or speaker_name,
        speaker_wav=None,
        speaking_rate=1.0,
        pause_ms=140,
        short_pause_ms=90,
        medium_pause_ms=140,
        final_pause_ms=180,
        pause_style="tight",
        clone_enabled=False,
        clone_ready=True,
        validation_status="fallback",
    )
    return {
        "jarvis_premium": premium,
        "jarvis_cinematic": cinematic,
        "custom_clone": custom,
        "fallback_basic": fallback,
        "jarvis_serious": replace(premium, name="jarvis_serious"),
        "assistant_neutral": replace(custom, name="assistant_neutral"),
    }


def resolve_profile_name(settings: Settings, profile_name: str | None = None) -> str:
    selected = (profile_name or settings.voice_clone_profile_default or settings.voice_profile_default or "jarvis_premium").strip()
    return selected or "jarvis_premium"


def _profile_fallbacks(settings: Settings, backend: str) -> tuple[str, ...]:
    ordered = [backend, *settings.voice_tts_provider_fallback_order]
    seen: set[str] = set()
    values: list[str] = []
    for name in ordered:
        if not name or name in seen:
            continue
        seen.add(name)
        values.append(name)
    return tuple(values[1:])


def _pause_values(pause_style: str) -> tuple[int, int, int]:
    normalized = (pause_style or "balanced").strip().casefold()
    if normalized == "tight":
        return 90, 170, 240
    if normalized == "cinematic":
        return 140, 260, 360
    return 120, 220, 320
