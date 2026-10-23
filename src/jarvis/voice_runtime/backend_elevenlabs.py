from __future__ import annotations

import io
import time
import wave
from typing import Any

import httpx

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.voice_runtime.base import SynthesisRequest, SynthesisResult
from jarvis.voice_runtime.buffers import wav_duration_seconds


def _pcm_to_wav(pcm: bytes, *, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw signed 16-bit little-endian PCM into a playable WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class ElevenLabsTTSProvider:
    """Optional cloud TTS quality tier; registers only with credentials.

    Requests raw PCM and wraps it into WAV locally so the existing winsound
    playback path keeps working without any new audio dependency.
    """

    provider_name = "elevenlabs"
    provider_kind = "cloud"
    _BASE_URL = "https://api.elevenlabs.io"

    def __init__(self, settings: Settings, *, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._client = client

    def _configured(self) -> bool:
        key = self._settings.elevenlabs_api_key
        return bool(
            key is not None
            and key.get_secret_value().strip()
            and self._settings.elevenlabs_voice_id.strip()
        )

    def health_check(self) -> dict[str, Any]:
        configured = self._configured()
        return {
            "provider_name": self.provider_name,
            "healthy": configured,
            "reason": None
            if configured
            else "faltan ELEVENLABS_API_KEY o ELEVENLABS_VOICE_ID",
        }

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not self._configured():
            raise JarvisError(
                "ElevenLabs no configurado: faltan ELEVENLABS_API_KEY o ELEVENLABS_VOICE_ID",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            )
        api_key = (self._settings.elevenlabs_api_key or "").get_secret_value()  # type: ignore[union-attr]
        started = time.perf_counter()
        url = f"{self._BASE_URL}/v1/text-to-speech/{self._settings.elevenlabs_voice_id}"
        try:
            client = self._client or httpx.Client(timeout=30.0)
            response = client.post(
                url,
                params={"output_format": "pcm_44100"},
                headers={"xi-api-key": api_key, "Accept": "audio/mpeg"},
                json={
                    "text": request.text,
                    "model_id": self._settings.elevenlabs_model,
                    "voice_settings": {
                        "stability": 0.45,
                        "similarity_boost": 0.8,
                        "style": 0.2,
                        "use_speaker_boost": True,
                    },
                },
            )
        except (httpx.HTTPError, OSError) as exc:
            raise JarvisError(
                f"ElevenLabs no disponible: {exc}",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            ) from exc
        if response.status_code != 200:
            detail = {
                401: "credencial rechazada",
                403: "credencial sin autorización para esta voz",
                422: "texto rechazado por la API",
                429: "cuota agotada o límite de tasa",
            }.get(
                response.status_code,
                f"HTTP {response.status_code}",
            )
            raise JarvisError(
                f"ElevenLabs rechazó la síntesis: {detail}",
                component="voice_runtime",
                code="tts_failed",
                recoverable=True,
            )
        wav_bytes = _pcm_to_wav(response.content, sample_rate=44_100)
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
