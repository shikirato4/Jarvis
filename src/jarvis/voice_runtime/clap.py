from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .base import AudioChunk, AudioEventType, ClapEvent
from .detection import detect_audio_activity


@dataclass(slots=True)
class ClapPatternDetector:
    sensitivity: float
    cooldown_seconds: float
    window_seconds: float
    _last_clap_at: float = 0.0
    _current_count: int = 0

    def process(self, chunk: AudioChunk, *, correlation_id: str | None = None) -> ClapEvent | None:
        detection = detect_audio_activity(
            chunk,
            silence_threshold=self.sensitivity / 4,
            speech_threshold=self.sensitivity,
        )
        now = monotonic()
        if detection.peak_energy < self.sensitivity:
            if self._current_count and (now - self._last_clap_at) > self.window_seconds:
                event = self._build_event(self._current_count, correlation_id=correlation_id)
                self._current_count = 0
                return event
            return None
        if self._last_clap_at and (now - self._last_clap_at) < self.cooldown_seconds:
            return None
        if self._last_clap_at and (now - self._last_clap_at) > self.window_seconds:
            finalized = self._build_event(self._current_count, correlation_id=correlation_id) if self._current_count else None
            self._current_count = 0
            self._last_clap_at = now
            self._current_count = 1
            return finalized
        self._last_clap_at = now
        self._current_count += 1
        if self._current_count >= 3:
            event = self._build_event(self._current_count, correlation_id=correlation_id)
            self._current_count = 0
            return event
        return None

    def flush(self, *, correlation_id: str | None = None) -> ClapEvent | None:
        if not self._current_count:
            return None
        event = self._build_event(self._current_count, correlation_id=correlation_id)
        self._current_count = 0
        return event

    @staticmethod
    def _build_event(clap_count: int, *, correlation_id: str | None = None) -> ClapEvent:
        mapping = {
            1: AudioEventType.ATTENTION,
            2: AudioEventType.LISTENING_START,
            3: AudioEventType.FULL_MODE,
        }
        return ClapEvent(
            event_type=mapping.get(clap_count, AudioEventType.CLAP),
            clap_count=clap_count,
            correlation_id=correlation_id,
            metadata={"pattern": clap_count},
        )
