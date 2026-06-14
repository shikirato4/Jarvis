from __future__ import annotations

import array
import io
import logging
import os
import time
import wave
from pathlib import Path
from typing import Any

from jarvis.core.errors import UIAutomationError

from .base import SynthesisRequest, SynthesisResult


class CoquiXTTSProvider:
    provider_name = "coqui_xtts"
    provider_kind = "local"

    def __init__(
        self,
        *,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        language: str = "es",
        speaker_wav: Path | None = None,
        speaker: str | None = None,
        device_preference: str = "auto",
        tos_agreed: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self._model_name = model_name
        self._default_language = language
        self._speaker_wav = speaker_wav
        self._speaker = speaker
        self._device_preference = device_preference
        self._tos_agreed = tos_agreed
        self._logger = logger or logging.getLogger("jarvis.voice.coqui_xtts")
        self._model = None
        self._device = "cpu"

    def health_check(self) -> dict[str, Any]:
        available, reason = self._dependency_status()
        if available:
            try:
                self._device = self._select_device()
            except Exception:
                self._device = "cpu"
        return {
            "provider_name": self.provider_name,
            "healthy": available,
            "model_name": self._model_name,
            "device": self._device,
            "reason": reason,
        }

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        started = time.perf_counter()
        text = (request.text or "").strip()
        if not text:
            raise UIAutomationError("coqui xtts requires non-empty text")
        try:
            model = self._ensure_model_loaded()
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"coqui xtts unavailable: {exc}") from exc

        language = request.language or self._default_language
        metadata = request.metadata or {}
        kwargs: dict[str, Any] = {}
        speaker_wav = metadata.get("speaker_wav")
        speaker_name = metadata.get("speaker_name")
        speaking_rate = metadata.get("speaking_rate")
        effective_speaker_wav = self._resolve_effective_speaker_wav(speaker_wav)
        if effective_speaker_wav is not None:
            kwargs["speaker_wav"] = str(effective_speaker_wav)
        if speaking_rate is not None:
            kwargs["speed"] = speaking_rate
        self._logger.info(
            "speaker_wav_effective",
            extra={
                "correlation_id": request.correlation_id,
                "profile_name": request.profile_name,
                "speaker_wav_requested": speaker_wav,
                "speaker_wav_effective": kwargs.get("speaker_wav"),
            },
        )
        if not kwargs.get("speaker_wav"):
            fallback_reason = "missing speaker_wav"
            self._logger.warning(
                "coqui_xtts_missing_speaker_wav",
                extra={
                    "correlation_id": request.correlation_id,
                    "language": language,
                    "profile_name": request.profile_name,
                    "speaker_name_requested": speaker_name,
                    "speaker_wav_requested": speaker_wav,
                    "speaker_wav_effective": None,
                    "fallback_reason": fallback_reason,
                },
            )
            raise UIAutomationError(fallback_reason)
        self._logger.info(
            "coqui_xtts_request",
            extra={
                "correlation_id": request.correlation_id,
                "language": language,
                "profile_name": request.profile_name,
                "style": metadata.get("style"),
                "speaker_name_requested": speaker_name,
                "speaker_wav_requested": speaker_wav,
                "speaker_name_effective": speaker_name or self._speaker,
                "speaker_wav_effective": kwargs.get("speaker_wav"),
                "speaking_rate_effective": kwargs.get("speed"),
                "rate_requested": request.rate,
            },
        )
        try:
            wav = model.tts(text=text, language=language, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "coqui_xtts_synthesis_failed",
                extra={
                    "correlation_id": request.correlation_id,
                    "speaker_name_effective": speaker_name or self._speaker,
                    "speaker_wav_effective": kwargs.get("speaker_wav"),
                    "speaking_rate_effective": kwargs.get("speed"),
                },
            )
            raise UIAutomationError(f"coqui xtts synthesis failed: {exc}") from exc
        audio_bytes = _floats_to_wav_bytes(wav)
        duration_seconds = _estimate_duration_seconds(wav)
        return SynthesisResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            audio_bytes=audio_bytes,
            audio_format="wav",
            text_payload=text,
            latency_ms=(time.perf_counter() - started) * 1000,
            duration_seconds=duration_seconds,
            metadata={
                **metadata,
                "model_name": self._model_name,
                "device": self._device,
                "language": language,
                "speaker": speaker_name or self._speaker,
                "speaker_name": speaker_name or self._speaker,
                "speaker_wav": kwargs.get("speaker_wav"),
                "speaker_wav_effective": kwargs.get("speaker_wav"),
                "voice_profile": request.profile_name,
                "speaking_rate": speaking_rate,
                "rate": request.rate,
                "style": metadata.get("style"),
            },
        )

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model
        if self._tos_agreed:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self._patch_torchaudio_load()
        try:
            from TTS.api import TTS  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"TTS package not installed: {exc}") from exc
        self._device = self._select_device()
        model = TTS(self._model_name)
        if hasattr(model, "to"):
            model = model.to(self._device)
        self._model = model
        self._logger.info("coqui_xtts_loaded", extra={"model_name": self._model_name, "device": self._device})
        return self._model

    def _select_device(self) -> str:
        preference = self._device_preference.casefold().strip()
        if preference in {"cpu", "cuda"}:
            return preference
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _dependency_status(self) -> tuple[bool, str]:
        try:
            from TTS.api import TTS  # type: ignore  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"TTS package unavailable: {exc}"
        if self._speaker_wav is not None and not self._speaker_wav.exists():
            return False, f"speaker_wav not found: {self._speaker_wav}"
        return True, "ok"

    def _resolve_effective_speaker_wav(self, speaker_wav: object) -> Path | None:
        candidate = speaker_wav or self._speaker_wav
        if candidate in (None, ""):
            return None
        path = Path(candidate).expanduser().resolve()
        if not path.exists():
            self._logger.warning("coqui_xtts_speaker_wav_not_found", extra={"speaker_wav_effective": str(path)})
            return None
        return path

    def _patch_torchaudio_load(self) -> None:
        try:
            import soundfile as sf  # type: ignore
            import torch  # type: ignore
            import torchaudio  # type: ignore
        except Exception:
            return

        load_fn = getattr(torchaudio, "load", None)
        if load_fn is None or getattr(load_fn, "_jarvis_soundfile_patch", False):
            return

        def _soundfile_load(path, *args, **kwargs):
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            tensor = torch.from_numpy(data.T)
            return tensor, sample_rate

        _soundfile_load._jarvis_soundfile_patch = True  # type: ignore[attr-defined]
        torchaudio.load = _soundfile_load


def _floats_to_wav_bytes(samples, *, sample_rate: int = 24000) -> bytes:
    pcm = array.array("h")
    for sample in samples:
        clipped = max(-1.0, min(1.0, float(sample)))
        pcm.append(int(clipped * 32767))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _estimate_duration_seconds(samples) -> float:
    try:
        return len(samples) / 24000
    except Exception:
        return 0.0
