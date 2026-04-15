from __future__ import annotations

import array
import io
import logging
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any

from jarvis.core.errors import ConfigurationError, UIAutomationError

from .base import (
    AudioCaptureRequest,
    AudioCaptureResult,
    AudioChunk,
    AudioInputBackend,
    AudioOutputBackend,
    CancellationToken,
    PlaybackHandle,
    STTProvider,
    SynthesisResult,
    TTSProvider,
    TranscriptionRequest,
    TranscriptionResult,
)


class AudioInputRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, AudioInputBackend] = {}

    def register(self, backend: AudioInputBackend) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(f"audio input backend '{backend.backend_name}' is already registered")
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str) -> AudioInputBackend | None:
        return self._backends.get(backend_name)

    def list_backends(self) -> list[AudioInputBackend]:
        return sorted(self._backends.values(), key=lambda item: item.backend_name)


class AudioOutputRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, AudioOutputBackend] = {}

    def register(self, backend: AudioOutputBackend) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(f"audio output backend '{backend.backend_name}' is already registered")
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str) -> AudioOutputBackend | None:
        return self._backends.get(backend_name)

    def list_backends(self) -> list[AudioOutputBackend]:
        return sorted(self._backends.values(), key=lambda item: item.backend_name)


class STTProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, STTProvider] = {}

    def register(self, provider: STTProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"stt provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> STTProvider | None:
        return self._providers.get(provider_name)

    def list_providers(self) -> list[STTProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class TTSProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, TTSProvider] = {}

    def register(self, provider: TTSProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"tts provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> TTSProvider | None:
        return self._providers.get(provider_name)

    def list_providers(self) -> list[TTSProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class LocalPlaybackHandle:
    def __init__(
        self,
        playback_id: str,
        cancellation: CancellationToken | None = None,
        *,
        duration_seconds: float | None = None,
        stop_callback=None,
    ) -> None:
        self.playback_id = playback_id
        self._cancellation = cancellation
        self._duration_seconds = duration_seconds
        self._stop_callback = stop_callback

    def stop(self) -> None:
        if self._cancellation is not None:
            self._cancellation.cancel()
        if self._stop_callback is not None:
            self._stop_callback()

    def wait(self, timeout_seconds: float | None = None) -> None:
        if self._duration_seconds is None:
            return
        deadline = time.perf_counter() + (timeout_seconds if timeout_seconds is not None else self._duration_seconds)
        target = time.perf_counter() + self._duration_seconds
        while time.perf_counter() < min(deadline, target):
            if self._cancellation is not None and self._cancellation.cancelled():
                return
            time.sleep(0.02)


class WavFileAudioInputBackend:
    backend_name = "wav_file"

    def health_check(self) -> dict[str, Any]:
        return {"backend_name": self.backend_name, "healthy": True}

    def capture(self, request: AudioCaptureRequest, *, cancellation: CancellationToken | None = None) -> AudioCaptureResult:
        if request.file_path is None:
            raise UIAutomationError("wav file backend requires a file_path")
        started = time.perf_counter()
        path = Path(request.file_path)
        with wave.open(str(path), "rb") as audio_file:
            frames = audio_file.readframes(audio_file.getnframes())
            duration = audio_file.getnframes() / audio_file.getframerate() if audio_file.getframerate() else 0
            chunk = AudioChunk(
                pcm_bytes=frames,
                sample_rate=audio_file.getframerate(),
                channels=audio_file.getnchannels(),
                sample_width_bytes=audio_file.getsampwidth(),
                duration_seconds=duration,
                source_name=str(path),
            )
        return AudioCaptureResult(
            backend_name=self.backend_name,
            chunks=[chunk],
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"file_path": str(path)},
        )


class SoundDeviceAudioInputBackend:
    backend_name = "sounddevice"

    def __init__(self) -> None:
        self._logger = logging.getLogger("jarvis.voice.sounddevice")

    def health_check(self) -> dict[str, Any]:
        try:
            import sounddevice  # type: ignore

            devices = sounddevice.query_devices()
            return {"backend_name": self.backend_name, "healthy": True, "devices": len(devices)}
        except Exception as exc:
            return {"backend_name": self.backend_name, "healthy": False, "error": str(exc)}

    def capture(self, request: AudioCaptureRequest, *, cancellation: CancellationToken | None = None) -> AudioCaptureResult:
        try:
            import numpy  # type: ignore
            import sounddevice  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"sounddevice backend unavailable: {exc}") from exc

        started = time.perf_counter()
        frames = int(request.duration_seconds * request.sample_rate)
        recording = sounddevice.rec(frames, samplerate=request.sample_rate, channels=request.channels, dtype="int16")
        sounddevice.wait()
        if cancellation is not None and cancellation.cancelled():
            raise UIAutomationError("audio capture cancelled")
        pcm = numpy.asarray(recording).astype("int16").tobytes()
        chunk = AudioChunk(
            pcm_bytes=pcm,
            sample_rate=request.sample_rate,
            channels=request.channels,
            sample_width_bytes=2,
            duration_seconds=request.duration_seconds,
            source_name=self.backend_name,
        )
        return AudioCaptureResult(
            backend_name=self.backend_name,
            chunks=[chunk],
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class InMemoryAudioInputBackend:
    backend_name = "in_memory"

    def __init__(self, chunks: list[AudioChunk] | None = None) -> None:
        self._chunks = list(chunks or [])

    def push_chunk(self, chunk: AudioChunk) -> None:
        self._chunks.append(chunk)

    def health_check(self) -> dict[str, Any]:
        return {"backend_name": self.backend_name, "healthy": True, "queued_chunks": len(self._chunks)}

    def capture(self, request: AudioCaptureRequest, *, cancellation: CancellationToken | None = None) -> AudioCaptureResult:
        started = time.perf_counter()
        chunks = list(self._chunks)
        self._chunks.clear()
        return AudioCaptureResult(
            backend_name=self.backend_name,
            chunks=chunks,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class InMemoryAudioOutputBackend:
    backend_name = "in_memory"

    def __init__(self) -> None:
        self.played: list[SynthesisResult] = []

    def health_check(self) -> dict[str, Any]:
        return {"backend_name": self.backend_name, "healthy": True, "played": len(self.played)}

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        self.played.append(result)
        return LocalPlaybackHandle(playback_id=str(uuid.uuid4()), cancellation=cancellation, duration_seconds=result.duration_seconds)


class WinsoundAudioOutputBackend:
    backend_name = "winsound"

    def __init__(self) -> None:
        self._active_path: str | None = None
        self._cleanup_timer: threading.Timer | None = None

    def health_check(self) -> dict[str, Any]:
        try:
            import winsound  # noqa: F401

            return {"backend_name": self.backend_name, "healthy": True}
        except Exception as exc:
            return {"backend_name": self.backend_name, "healthy": False, "error": str(exc)}

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        try:
            import winsound
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"winsound backend unavailable: {exc}") from exc
        wav_bytes = result.audio_bytes
        if not wav_bytes:
            raise UIAutomationError("winsound backend requires wav audio bytes")
        if result.audio_format not in {None, "", "wav"}:
            raise UIAutomationError(f"winsound backend only supports wav audio, got {result.audio_format}")
        temp_file = tempfile.NamedTemporaryFile(prefix="jarvis-voice-", suffix=".wav", delete=False)
        temp_file.write(wav_bytes)
        temp_file.flush()
        temp_file.close()
        self._active_path = temp_file.name
        winsound.PlaySound(self._active_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        self._schedule_cleanup(result.duration_seconds, self._active_path)
        return LocalPlaybackHandle(
            playback_id=str(uuid.uuid4()),
            cancellation=cancellation,
            duration_seconds=result.duration_seconds,
            stop_callback=self._stop_active_playback,
        )

    def _schedule_cleanup(self, duration_seconds: float | None, audio_path: str) -> None:
        delay = max((duration_seconds or 0.5) + 0.5, 1.0)
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
        self._cleanup_timer = threading.Timer(delay, lambda: self._delete_file(audio_path))
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _stop_active_playback(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, 0)
        except Exception:
            pass
        if self._active_path is not None:
            self._delete_file(self._active_path)
            self._active_path = None

    @staticmethod
    def _delete_file(audio_path: str) -> None:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except Exception:
            pass


class Pyttsx3AudioOutputBackend:
    backend_name = "pyttsx3"

    def __init__(self) -> None:
        self._engine = None
        self._engine_lock = threading.Lock()

    def health_check(self) -> dict[str, Any]:
        try:
            self._ensure_engine()
            return {"backend_name": self.backend_name, "healthy": True}
        except Exception as exc:
            return {"backend_name": self.backend_name, "healthy": False, "error": str(exc)}

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        try:
            engine = self._ensure_engine()
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"pyttsx3 output backend unavailable: {exc}") from exc
        text = result.text_payload or ""
        rate = result.metadata.get("rate")
        voice_name = result.metadata.get("voice_name")
        if rate is not None:
            engine.setProperty("rate", rate)
        if voice_name:
            voices = engine.getProperty("voices")
            for voice in voices:
                if str(voice_name).casefold() in str(getattr(voice, "name", "")).casefold():
                    engine.setProperty("voice", voice.id)
                    break
        if cancellation is not None and cancellation.cancelled():
            return LocalPlaybackHandle(playback_id=str(uuid.uuid4()), cancellation=cancellation)
        engine.say(text)
        engine.runAndWait()
        return LocalPlaybackHandle(playback_id=str(uuid.uuid4()), cancellation=cancellation, duration_seconds=result.duration_seconds)

    def _ensure_engine(self):
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            return self._engine


class FasterWhisperSTTProvider:
    provider_name = "faster_whisper"
    provider_kind = "local"

    def __init__(self, *, model_name: str = "base") -> None:
        self._model_name = model_name
        self._model = None

    def health_check(self) -> dict[str, Any]:
        try:
            self._ensure_model_loaded()
            return {"provider_name": self.provider_name, "healthy": True}
        except Exception as exc:
            return {"provider_name": self.provider_name, "healthy": False, "error": str(exc)}

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        try:
            model = self._ensure_model_loaded()
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"faster-whisper unavailable: {exc}") from exc
        started = time.perf_counter()
        audio_source = request.file_path or _chunks_to_wav_bytes(request.audio_chunks)
        segments, info = model.transcribe(audio_source, language=request.language, initial_prompt=request.prompt)
        parsed_segments = list(segments)
        text = " ".join(segment.text.strip() for segment in parsed_segments).strip()
        return TranscriptionResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text=text,
            language=getattr(info, "language", request.language),
            latency_ms=(time.perf_counter() - started) * 1000,
            confidence=getattr(info, "language_probability", None),
            segments=[
                {
                    "text": segment.text.strip(),
                    "start_seconds": segment.start,
                    "end_seconds": segment.end,
                }
                for segment in parsed_segments
            ],
        )

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(str(exc)) from exc
        self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")
        return self._model


class InMemorySTTProvider:
    provider_name = "in_memory"
    provider_kind = "local"

    def __init__(self, *, text: str = "", fail: bool = False) -> None:
        self._text = text
        self._fail = fail

    def health_check(self) -> dict[str, Any]:
        return {"provider_name": self.provider_name, "healthy": not self._fail}

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        if self._fail:
            raise UIAutomationError("in-memory stt failed")
        text = request.metadata.get("mock_text", self._text)
        if not text and request.audio_chunks:
            for chunk in request.audio_chunks:
                mock_text = chunk.metadata.get("mock_text")
                if mock_text:
                    text = str(mock_text)
                    break
        return TranscriptionResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text=str(text),
            latency_ms=1.0,
        )


class InMemoryTTSProvider:
    provider_name = "in_memory"
    provider_kind = "local"

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def health_check(self) -> dict[str, Any]:
        return {"provider_name": self.provider_name, "healthy": not self._fail}

    def synthesize(self, request) -> SynthesisResult:
        if self._fail:
            raise UIAutomationError("in-memory tts failed")
        return SynthesisResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text_payload=request.text,
            latency_ms=1.0,
            duration_seconds=min(len(request.text) / 15, 30),
            audio_format="text",
        )


class Pyttsx3TTSProvider:
    provider_name = "pyttsx3"
    provider_kind = "local"

    def __init__(self) -> None:
        self._engine = None
        self._engine_lock = threading.Lock()

    def health_check(self) -> dict[str, Any]:
        try:
            self._ensure_engine()
            return {"provider_name": self.provider_name, "healthy": True}
        except Exception as exc:
            return {"provider_name": self.provider_name, "healthy": False, "error": str(exc)}

    def synthesize(self, request) -> SynthesisResult:
        try:
            engine = self._ensure_engine()
            if request.rate is not None:
                engine.setProperty("rate", request.rate)
            if request.voice_name:
                voices = engine.getProperty("voices")
                for voice in voices:
                    if request.voice_name.casefold() in str(getattr(voice, "name", "")).casefold():
                        engine.setProperty("voice", voice.id)
                        break
        except Exception as exc:  # noqa: BLE001
            raise UIAutomationError(f"pyttsx3 tts unavailable: {exc}") from exc
        return SynthesisResult(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            text_payload=request.text,
            latency_ms=1.0,
            duration_seconds=min(len(request.text) / 14, 30),
            audio_format="text",
            metadata={
                "voice_name": request.voice_name,
                "rate": request.rate,
                "voice_profile": request.profile_name,
                **request.metadata,
            },
        )

    def _ensure_engine(self):
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            return self._engine


def _chunks_to_wav_bytes(chunks: list[AudioChunk]) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        sample_rate = chunks[0].sample_rate if chunks else 16_000
        channels = chunks[0].channels if chunks else 1
        sample_width = chunks[0].sample_width_bytes if chunks else 2
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(chunk.pcm_bytes for chunk in chunks))
    buffer.seek(0)
    return buffer
