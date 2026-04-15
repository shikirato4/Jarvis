from __future__ import annotations

from jarvis.config import Settings
from jarvis.core.errors import VoiceValidationError
from jarvis.core.modes import ModeManager


def validate_microphone_access(mode_manager: ModeManager) -> None:
    if not mode_manager.current_policy().microphone_allowed:
        raise VoiceValidationError("microphone access is not allowed in the current mode", details={"mode": mode_manager.current_mode().value})


def validate_playback_access(mode_manager: ModeManager) -> None:
    if not mode_manager.current_policy().playback_allowed:
        raise VoiceValidationError("voice playback is not allowed in the current mode", details={"mode": mode_manager.current_mode().value})


def validate_clap_access(mode_manager: ModeManager) -> None:
    if not mode_manager.current_policy().clap_detection_allowed:
        raise VoiceValidationError("clap detection is not allowed in the current mode", details={"mode": mode_manager.current_mode().value})


def validate_session_duration(settings: Settings, duration_seconds: float | None) -> None:
    if duration_seconds is None:
        return
    if duration_seconds <= 0 or duration_seconds > settings.voice_max_session_seconds:
        raise VoiceValidationError(
            "voice session duration exceeds configured limits",
            details={"duration_seconds": duration_seconds, "max_seconds": settings.voice_max_session_seconds},
        )
