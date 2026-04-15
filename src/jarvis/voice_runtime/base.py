from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from threading import Event
from typing import Any, Protocol

from pydantic import Field as PydanticField

from jarvis.models.base import JarvisBaseModel


class VoiceSessionState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    DICTATING = "dictating"
    SPEAKING = "speaking"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"
    FAILED = "failed"


class VoiceSessionMode(StrEnum):
    COMMAND = "command"
    LISTEN = "listen"
    DICTATION = "dictation"


class AudioEventType(StrEnum):
    ATTENTION = "attention"
    LISTENING_START = "listening_start"
    FULL_MODE = "full_mode"
    CLAP = "clap"
    SILENCE = "silence"
    SPEECH = "speech"
    CANCEL = "cancel"
    VOICE_COMMAND = "voice_command"


class AudioChunk(JarvisBaseModel):
    pcm_bytes: bytes = b""
    sample_rate: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2
    duration_seconds: float = 0.0
    source_name: str | None = None
    captured_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class AudioBufferWindow(JarvisBaseModel):
    chunks: list[AudioChunk] = PydanticField(default_factory=list)
    total_duration_seconds: float = 0.0


class AudioCaptureRequest(JarvisBaseModel):
    duration_seconds: float = 3.0
    sample_rate: int = 16_000
    channels: int = 1
    chunk_duration_seconds: float = 1.0
    file_path: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class AudioCaptureResult(JarvisBaseModel):
    backend_name: str
    chunks: list[AudioChunk] = PydanticField(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class TranscriptionRequest(JarvisBaseModel):
    audio_chunks: list[AudioChunk] = PydanticField(default_factory=list)
    file_path: str | None = None
    language: str | None = None
    prompt: str | None = None
    task_type: str = "transcription"
    correlation_id: str | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class STTTranscriptSegment(JarvisBaseModel):
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    confidence: float | None = None


class TranscriptionResult(JarvisBaseModel):
    provider_name: str
    provider_kind: str
    text: str
    language: str | None = None
    latency_ms: float
    confidence: float | None = None
    fallback_used: bool = False
    segments: list[STTTranscriptSegment] = PydanticField(default_factory=list)
    created_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class SynthesisRequest(JarvisBaseModel):
    text: str
    voice_name: str | None = None
    profile_name: str | None = None
    language: str | None = None
    rate: int | None = None
    play: bool = True
    correlation_id: str | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class SynthesisResult(JarvisBaseModel):
    provider_name: str
    provider_kind: str
    backend_name: str | None = None
    played: bool = False
    audio_bytes: bytes | None = None
    audio_path: str | None = None
    text_payload: str | None = None
    audio_format: str | None = None
    latency_ms: float = 0.0
    duration_seconds: float | None = None
    fallback_used: bool = False
    created_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class VoiceDetectionResult(JarvisBaseModel):
    speech_detected: bool = False
    silence_detected: bool = False
    average_energy: float = 0.0
    peak_energy: float = 0.0
    clap_count: int = 0
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class AudioEvent(JarvisBaseModel):
    event_type: AudioEventType
    confidence: float = 1.0
    correlation_id: str | None = None
    created_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class ClapEvent(AudioEvent):
    clap_count: int = 1


class VoiceCommandEvent(AudioEvent):
    transcript: str


class VoicePolicy(JarvisBaseModel):
    mode: str
    microphone_allowed: bool = False
    playback_allowed: bool = False
    clap_detection_allowed: bool = False
    voice_streaming_allowed: bool = False
    allowed_voice_provider_kinds: tuple[str, ...] = ()


class VoiceSession(JarvisBaseModel):
    session_id: str
    correlation_id: str
    mode: VoiceSessionMode
    state: VoiceSessionState
    transcripts: list[str] = PydanticField(default_factory=list)
    events: list[AudioEvent] = PydanticField(default_factory=list)
    last_error: str | None = None
    started_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class VoiceSessionRequest(JarvisBaseModel):
    mode: VoiceSessionMode = VoiceSessionMode.LISTEN
    duration_seconds: float | None = None
    language: str | None = None
    target_window: str | None = None
    ui_mode: str | None = None
    playback_response: bool = False
    correlation_id: str | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)


class VoiceOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    success: bool
    message: str
    session_id: str | None = None
    latency_ms: float | None = None
    data: dict[str, Any] = PydanticField(default_factory=dict)
    started_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = PydanticField(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class CancellationToken:
    correlation_id: str
    _event: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()


class PlaybackHandle(Protocol):
    playback_id: str

    def stop(self) -> None: ...

    def wait(self, timeout_seconds: float | None = None) -> None: ...


class STTProvider(Protocol):
    provider_name: str
    provider_kind: str

    def health_check(self) -> dict[str, Any]: ...

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult: ...


class TTSProvider(Protocol):
    provider_name: str
    provider_kind: str

    def health_check(self) -> dict[str, Any]: ...

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult: ...


class AudioInputBackend(Protocol):
    backend_name: str

    def health_check(self) -> dict[str, Any]: ...

    def capture(self, request: AudioCaptureRequest, *, cancellation: CancellationToken | None = None) -> AudioCaptureResult: ...


class AudioOutputBackend(Protocol):
    backend_name: str

    def health_check(self) -> dict[str, Any]: ...

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle: ...
