from __future__ import annotations

from typing import Any


class JarvisError(Exception):
    """Base application error with structured metadata."""

    default_code = "jarvis_error"
    default_component = "runtime"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        component: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.component = component or self.default_component
        self.details = details or {}
        self.recoverable = recoverable

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "code": self.code,
            "component": self.component,
            "details": self.details,
            "recoverable": self.recoverable,
        }


class ConfigurationError(JarvisError):
    """Raised when configuration is invalid."""

    default_code = "configuration_error"
    default_component = "config"


class SafetyViolationError(JarvisError):
    """Raised when an operation violates a safety policy."""

    default_code = "safety_violation"
    default_component = "safety"


class ActionNotFoundError(JarvisError):
    """Raised when an action is not registered."""

    default_code = "action_not_found"
    default_component = "actions"


class ActionValidationError(JarvisError):
    """Raised when an action payload is invalid."""

    default_code = "action_validation_error"
    default_component = "actions"


class ActionExecutionError(JarvisError):
    """Raised when an action execution fails."""

    default_code = "action_execution_error"
    default_component = "actions"


class RollbackError(JarvisError):
    """Raised when an action rollback fails."""

    default_code = "rollback_error"
    default_component = "actions"


class PersistenceError(JarvisError):
    """Raised when persistence operations fail."""

    default_code = "persistence_error"
    default_component = "memory"


class CapabilityUnavailableError(JarvisError):
    """Raised when a requested capability is not available."""

    default_code = "capability_unavailable"
    default_component = "capability"


class OrchestrationError(JarvisError):
    """Raised when the orchestration pipeline cannot fulfill a request."""

    default_code = "orchestration_error"
    default_component = "cognition"


class ModeTransitionError(JarvisError):
    """Raised when the runtime mode cannot transition safely."""

    default_code = "mode_transition_error"
    default_component = "modes"


class MetaCommandParseError(JarvisError):
    """Raised when a metacommand is invalid."""

    default_code = "metacommand_parse_error"
    default_component = "metacommands"


class ToolNotFoundError(JarvisError):
    """Raised when a tool is not registered."""

    default_code = "tool_not_found"
    default_component = "tools"


class ToolValidationError(JarvisError):
    """Raised when a tool payload is invalid."""

    default_code = "tool_validation_error"
    default_component = "tools"


class ToolExecutionError(JarvisError):
    """Raised when a tool invocation fails."""

    default_code = "tool_execution_error"
    default_component = "tools"


class TaskRoutingError(JarvisError):
    """Raised when a task cannot be routed."""

    default_code = "task_routing_error"
    default_component = "routing"


class ServiceUnavailableError(JarvisError):
    """Raised when a service facade is not available."""

    default_code = "service_unavailable"
    default_component = "services"


class ModelProviderError(JarvisError):
    """Raised when a model provider fails."""

    default_code = "model_provider_error"
    default_component = "models"


class ModelRoutingError(JarvisError):
    """Raised when a model route cannot be resolved."""

    default_code = "model_routing_error"
    default_component = "models"


class EmbeddingProviderError(JarvisError):
    """Raised when an embedding provider fails."""

    default_code = "embedding_provider_error"
    default_component = "semantic_memory"


class EmbeddingRoutingError(JarvisError):
    """Raised when an embedding route cannot be resolved."""

    default_code = "embedding_routing_error"
    default_component = "semantic_memory"


class UIAutomationError(JarvisError):
    """Raised when UI automation fails."""

    default_code = "ui_automation_error"
    default_component = "ui_automation"


class UIValidationError(JarvisError):
    """Raised when a UI automation request is invalid or unsafe."""

    default_code = "ui_validation_error"
    default_component = "ui_automation"


class UICancelledError(JarvisError):
    """Raised when a UI automation operation is cancelled."""

    default_code = "ui_cancelled"
    default_component = "ui_automation"


class VoiceRuntimeError(JarvisError):
    """Raised when the voice runtime fails."""

    default_code = "voice_runtime_error"
    default_component = "voice_runtime"


class VoiceValidationError(JarvisError):
    """Raised when a voice request is invalid or unsafe."""

    default_code = "voice_validation_error"
    default_component = "voice_runtime"


class VoiceCancelledError(JarvisError):
    """Raised when a voice operation is cancelled."""

    default_code = "voice_cancelled"
    default_component = "voice_runtime"


class VisionRuntimeError(JarvisError):
    """Raised when the vision runtime fails."""

    default_code = "vision_runtime_error"
    default_component = "vision_runtime"


class VisionValidationError(JarvisError):
    """Raised when a vision request is invalid or unsafe."""

    default_code = "vision_validation_error"
    default_component = "vision_runtime"


class WritingRuntimeError(JarvisError):
    """Raised when the writing runtime fails."""

    default_code = "writing_runtime_error"
    default_component = "writing_runtime"


class AutonomyRuntimeError(JarvisError):
    """Raised when the autonomy runtime fails."""

    default_code = "autonomy_runtime_error"
    default_component = "autonomy"


class AutonomyValidationError(JarvisError):
    """Raised when an autonomy mission or step is invalid."""

    default_code = "autonomy_validation_error"
    default_component = "autonomy"


class SystemRuntimeError(JarvisError):
    """Raised when the system runtime fails."""

    default_code = "system_runtime_error"
    default_component = "system_runtime"


class SystemResolutionError(JarvisError):
    """Raised when a system target cannot be resolved safely."""

    default_code = "system_resolution_error"
    default_component = "system_runtime"


class SystemLaunchError(JarvisError):
    """Raised when a resolved system target cannot be opened or launched."""

    default_code = "system_launch_error"
    default_component = "system_runtime"


class SystemSafetyError(JarvisError):
    """Raised when a system operation violates system-runtime policy."""

    default_code = "system_safety_error"
    default_component = "system_runtime"


class UnityRuntimeError(JarvisError):
    """Raised when the Unity runtime fails."""

    default_code = "unity_runtime_error"
    default_component = "unity_runtime"


class UnityProjectResolutionError(JarvisError):
    """Raised when a Unity project cannot be resolved safely."""

    default_code = "unity_project_resolution_error"
    default_component = "unity_runtime"


class UnityEditorOperationError(JarvisError):
    """Raised when a Unity editor operation cannot be prepared or executed."""

    default_code = "unity_editor_operation_error"
    default_component = "unity_runtime"


class UnityBridgeError(JarvisError):
    """Raised when a Unity bridge operation fails."""

    default_code = "unity_bridge_error"
    default_component = "unity_runtime"


class UnitySafetyError(JarvisError):
    """Raised when a Unity operation violates Unity-runtime safety policy."""

    default_code = "unity_safety_error"
    default_component = "unity_runtime"
