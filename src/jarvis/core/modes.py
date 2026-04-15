from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from jarvis.models.base import JarvisBaseModel

from .errors import ModeTransitionError, SafetyViolationError


class ExecutionMode(StrEnum):
    STANDBY = "standby"
    ASSIST = "assist"
    RESEARCH = "research"
    OPERATOR = "operator"
    AUTOMATION = "automation"


class ModePolicy(JarvisBaseModel):
    mode: ExecutionMode
    description: str
    default_intent: str | None = None
    allowed_action_prefixes: tuple[str, ...]
    allowed_tool_tags: tuple[str, ...] = ()
    allowed_provider_kinds: tuple[str, ...] = ()
    allowed_voice_provider_kinds: tuple[str, ...] = ()
    network_allowed: bool = False
    streaming_allowed: bool = False
    microphone_allowed: bool = False
    playback_allowed: bool = False
    clap_detection_allowed: bool = False
    voice_streaming_allowed: bool = False
    allow_metacommands: bool = True


class ModeManager:
    def __init__(self, default_mode: ExecutionMode | str = ExecutionMode.ASSIST) -> None:
        self._policies = self._build_policies()
        self._current_mode = ExecutionMode(default_mode)
        self._previous_mode: ExecutionMode | None = None
        self._sticky = True
        self._changed_at = datetime.now(timezone.utc)
        self._reason = "bootstrap"

    def current_mode(self) -> ExecutionMode:
        return self._current_mode

    def current_policy(self) -> ModePolicy:
        return self._policies[self._current_mode]

    def set_mode(self, mode: ExecutionMode | str, *, reason: str | None = None, sticky: bool = True) -> ExecutionMode:
        target_mode = ExecutionMode(mode)
        if target_mode not in self._policies:
            raise ModeTransitionError(f"mode '{target_mode}' is not configured")
        self._previous_mode = self._current_mode
        self._current_mode = target_mode
        self._sticky = sticky
        self._changed_at = datetime.now(timezone.utc)
        self._reason = reason
        return self._current_mode

    def snapshot(self) -> "ModeSnapshot":
        from .models import ModeSnapshot

        return ModeSnapshot(
            active_mode=self._current_mode.value,
            previous_mode=self._previous_mode.value if self._previous_mode else None,
            sticky=self._sticky,
            changed_at=self._changed_at,
            reason=self._reason,
        )

    def resolve_intent(self, requested_intent: str | None) -> str | None:
        if requested_intent:
            return requested_intent
        return self.current_policy().default_intent

    def validate_action(self, action_name: str) -> None:
        allowed_prefixes = self.current_policy().allowed_action_prefixes
        if any(action_name.startswith(prefix) for prefix in allowed_prefixes):
            return
        raise SafetyViolationError(
            f"action '{action_name}' is not allowed while mode is '{self._current_mode.value}'",
            details={"mode": self._current_mode.value, "action_name": action_name},
        )

    def validate_tool_tags(self, tool_name: str, tags: tuple[str, ...]) -> None:
        allowed_tags = self.current_policy().allowed_tool_tags
        if not allowed_tags:
            return
        if any(tag in allowed_tags for tag in tags):
            return
        raise SafetyViolationError(
            f"tool '{tool_name}' is not allowed while mode is '{self._current_mode.value}'",
            details={"mode": self._current_mode.value, "tool_name": tool_name},
        )

    @staticmethod
    def _build_policies() -> dict[ExecutionMode, ModePolicy]:
        return {
            ExecutionMode.STANDBY: ModePolicy(
                mode=ExecutionMode.STANDBY,
                description="Minimal mode for inspection and state access.",
                default_intent=None,
                allowed_action_prefixes=("memory.",),
                allowed_tool_tags=("state", "memory"),
                allowed_provider_kinds=("local", "remote"),
                allowed_voice_provider_kinds=("local",),
                network_allowed=False,
                streaming_allowed=False,
                microphone_allowed=False,
                playback_allowed=False,
                clap_detection_allowed=False,
                voice_streaming_allowed=False,
            ),
            ExecutionMode.ASSIST: ModePolicy(
                mode=ExecutionMode.ASSIST,
                description="Default conversational and coordination mode.",
                default_intent=None,
                allowed_action_prefixes=("memory.", "research.", "writer.", "vision.", "voice."),
                allowed_tool_tags=("memory", "research", "writer", "vision", "voice", "state", "models", "reasoning"),
                allowed_provider_kinds=("local", "remote"),
                allowed_voice_provider_kinds=("local",),
                network_allowed=False,
                streaming_allowed=False,
                microphone_allowed=True,
                playback_allowed=True,
                clap_detection_allowed=True,
                voice_streaming_allowed=False,
            ),
            ExecutionMode.RESEARCH: ModePolicy(
                mode=ExecutionMode.RESEARCH,
                description="Bias routing toward evidence collection and briefs.",
                default_intent="research",
                allowed_action_prefixes=("memory.", "research.", "writer.", "vision.", "voice."),
                allowed_tool_tags=("memory", "research", "writer", "vision", "voice", "state", "models", "reasoning"),
                allowed_provider_kinds=("local", "remote"),
                allowed_voice_provider_kinds=("local",),
                network_allowed=False,
                streaming_allowed=False,
                microphone_allowed=True,
                playback_allowed=True,
                clap_detection_allowed=True,
                voice_streaming_allowed=False,
            ),
            ExecutionMode.OPERATOR: ModePolicy(
                mode=ExecutionMode.OPERATOR,
                description="Operational control mode with command execution enabled.",
                default_intent="operate",
                allowed_action_prefixes=("memory.", "research.", "writer.", "vision.", "voice.", "operations.", "interface."),
                allowed_tool_tags=("memory", "research", "writer", "vision", "voice", "operations", "interface", "state", "models", "reasoning"),
                allowed_provider_kinds=("local", "remote"),
                allowed_voice_provider_kinds=("local",),
                network_allowed=False,
                streaming_allowed=False,
                microphone_allowed=True,
                playback_allowed=True,
                clap_detection_allowed=True,
                voice_streaming_allowed=True,
            ),
            ExecutionMode.AUTOMATION: ModePolicy(
                mode=ExecutionMode.AUTOMATION,
                description="Scheduled flows and background coordination mode.",
                default_intent=None,
                allowed_action_prefixes=("memory.", "research.", "writer.", "vision.", "voice.", "operations.", "interface."),
                allowed_tool_tags=("memory", "research", "writer", "vision", "voice", "operations", "interface", "state", "models", "reasoning"),
                allowed_provider_kinds=("local", "remote"),
                allowed_voice_provider_kinds=("local",),
                network_allowed=False,
                streaming_allowed=False,
                microphone_allowed=True,
                playback_allowed=True,
                clap_detection_allowed=True,
                voice_streaming_allowed=True,
            ),
        }
