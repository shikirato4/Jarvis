from __future__ import annotations

from .base import AudioEvent, AudioEventType, ClapEvent, VoiceCommandEvent


def map_clap_to_event(event: ClapEvent) -> AudioEvent:
    return AudioEvent(
        event_type=event.event_type,
        confidence=event.confidence,
        correlation_id=event.correlation_id,
        metadata={**event.metadata, "clap_count": event.clap_count},
    )


def build_cancel_event(*, transcript: str, correlation_id: str | None = None) -> VoiceCommandEvent:
    return VoiceCommandEvent(
        event_type=AudioEventType.CANCEL,
        transcript=transcript,
        correlation_id=correlation_id,
        metadata={"cancel_phrase": True},
    )
