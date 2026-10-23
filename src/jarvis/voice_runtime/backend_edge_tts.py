from __future__ import annotations

import asyncio
import io
import time
from typing import Any
from collections.abc import Callable

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.voice_runtime.backend_elevenlabs import _pcm_to_wav
from jarvis.voice_runtime.base import SynthesisRequest, SynthesisResult
from jarvis.voice_runtime.buffers import wav_duration_seconds


def _decode_mp3_to_pcm(mp3: bytes) -> tuple[bytes, int]:
    """Decode MP3 to signed 16-bit little-endian PCM via libsndfile."""
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(mp3), dtype="int16")
    return data.tobytes(), int(sample_rate)


class EdgeTTSProvider:
    """Free unlimited neural TTS via Microsoft Edge's Read-Aloud endpoint.

    400+ neural voices across 100+ languages, no API key, no quota. The
    stream returns MP3, which is decoded locally and wrapped into WAV so
    the existing winsound playback keeps working.
    """

    provider_name = "edge_tts"
    provider_kind = "cloud"

    def __init__(
        self,
        settings: Settings,
        *,
        communicate_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._settings = settings
        self._communicate_factory = communicate_factory

    def _communicate(self, text: str, voice: str) -> Any:
        if self._communicate_factory is not None:
            return self._communicate_factory(text, voice)
        import edge_tts

        return edge_tts.Communicate(text, voice)

    def health_check(self) -> dict[str, Any]:
        enabled = bool(self._settings.voice_edge_tts_enabled)
        return {
            "provider_name": self.provider_name,
            "healthy": enabled,
            "reason": None if enabled else "VOICE_EDGE_TTS_ENABLED esta desactivado",
        }

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not self._settings.voice_edge_tts_enabled:
            raise JarvisError(
                "Edge TTS deshabilitado: activa VOICE_EDGE_TTS_ENABLED=true en .env",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            )
        started = time.perf_counter()
        communicate = self._communicate(
            request.text,
            self._settings.voice_edge_tts_voice,
        )

        async def _collect() -> bytes:
            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    buffer.extend(chunk["data"])
            return bytes(buffer)

        try:
            mp3 = asyncio.run(_collect())
        except Exception as exc:  # noqa: BLE001 - map transport failures cleanly
            raise JarvisError(
                f"Edge TTS no disponible: {exc}",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            ) from exc
        if not mp3:
            raise JarvisError(
                "Edge TTS no devolvio audio",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            )
        pcm, sample_rate = _decode_mp3_to_pcm(mp3)
        wav_bytes = _pcm_to_wav(pcm, sample_rate=sample_rate)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return SynthesisResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            played=False,
            audio_bytes=wav_bytes,
            audio_format="wav",
            duration_seconds=wav_duration_seconds(wav_bytes),
            text_payload=request.text,
            latency_ms=latency_ms,
        )
