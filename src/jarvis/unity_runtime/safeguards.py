from __future__ import annotations

from pathlib import Path

from pydantic import Field

from jarvis.core.errors import UnitySafetyError
from jarvis.models.base import JarvisBaseModel

from .base import UnityControlAdvice, UnityEditorOperationKind, UnityProjectDescriptor, UnityTargetSensitivity


class UnitySafetyPolicy(JarvisBaseModel):
    allowed_project_roots: list[str] = Field(default_factory=list)
    blocked_project_roots: list[str] = Field(default_factory=list)
    allowed_installation_paths: list[str] = Field(default_factory=list)
    require_confirmation_for_project_creation: bool = True
    require_confirmation_for_script_overwrite: bool = True
    require_confirmation_for_editor_open: bool = True
    require_confirmation_for_launch: bool = True
    require_confirmation_for_bridge_commands: bool = True
    require_confirmation_for_editor_commands: bool = True
    require_confirmation_for_custom_commands: bool = True
    blocked_asset_paths: list[str] = Field(default_factory=list)
    blocked_extensions: list[str] = Field(default_factory=list)
    allowed_bridge_commands: list[str] = Field(default_factory=list)
    blocked_bridge_commands: list[str] = Field(default_factory=list)
    allow_stub_when_bridge_unavailable: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)


def build_unity_safety_policy(settings) -> UnitySafetyPolicy:
    return UnitySafetyPolicy(
        allowed_project_roots=[str(path) for path in settings.resolved_unity_project_allowed_roots],
        blocked_project_roots=[str(path) for path in settings.resolved_unity_project_blocked_roots],
        allowed_installation_paths=[str(path) for path in settings.resolved_unity_allowed_installation_paths],
        require_confirmation_for_project_creation=settings.unity_require_confirmation_for_project_creation,
        require_confirmation_for_script_overwrite=settings.unity_require_confirmation_for_script_overwrite,
        require_confirmation_for_editor_open=settings.unity_require_confirmation_for_editor_open,
        require_confirmation_for_launch=settings.unity_require_confirmation_for_launch,
        require_confirmation_for_bridge_commands=settings.unity_require_confirmation_for_bridge_commands,
        require_confirmation_for_editor_commands=settings.unity_require_confirmation_for_editor_commands,
        require_confirmation_for_custom_commands=settings.unity_require_confirmation_for_custom_commands,
        blocked_asset_paths=[str(path) for path in settings.resolved_unity_blocked_asset_paths],
        blocked_extensions=[item.casefold() for item in settings.unity_blocked_extensions],
        allowed_bridge_commands=[item.strip() for item in settings.unity_allowed_bridge_commands],
        blocked_bridge_commands=[item.strip() for item in settings.unity_blocked_bridge_commands],
        allow_stub_when_bridge_unavailable=settings.unity_allow_stub_when_bridge_unavailable,
    )


def validate_project_allowed(project: UnityProjectDescriptor, *, policy: UnitySafetyPolicy) -> None:
    root = Path(project.project_root).resolve(strict=False)
    if policy.allowed_project_roots and not any(root == candidate or root.is_relative_to(candidate) for candidate in map(_resolve, policy.allowed_project_roots)):
        raise UnitySafetyError(f"unity project '{root}' is outside allowed roots")
    if any(root == candidate or root.is_relative_to(candidate) for candidate in map(_resolve, policy.blocked_project_roots)):
        raise UnitySafetyError(f"unity project '{root}' is blocked")


def validate_asset_write_path(project: UnityProjectDescriptor, path: Path, *, policy: UnitySafetyPolicy) -> None:
    root = Path(project.project_root).resolve(strict=False)
    assets_root = (root / "Assets").resolve(strict=False)
    resolved = path.resolve(strict=False)
    if not (resolved == assets_root or resolved.is_relative_to(assets_root) or resolved.parent == assets_root or resolved.parent.is_relative_to(assets_root)):
        raise UnitySafetyError("unity asset writes must stay inside Assets/")
    if any(resolved == candidate or resolved.is_relative_to(candidate) for candidate in map(_resolve, policy.blocked_asset_paths)):
        raise UnitySafetyError(f"unity asset path '{resolved}' is blocked")
    if resolved.suffix.casefold() in set(policy.blocked_extensions):
        raise UnitySafetyError(f"unity asset extension '{resolved.suffix}' is blocked")


def control_advice_for_operation(
    *,
    operation_kind: UnityEditorOperationKind,
    project: UnityProjectDescriptor | None,
    policy: UnitySafetyPolicy | None = None,
    would_overwrite: bool = False,
    writes_project_settings: bool = False,
    uses_bridge: bool = False,
) -> UnityControlAdvice:
    warnings: list[str] = []
    tags: list[str] = [f"operation:{operation_kind.value}"]
    sensitivity = UnityTargetSensitivity.LOW
    requires_confirmation = False
    reason: str | None = None

    if operation_kind == UnityEditorOperationKind.CREATE_PROJECT:
        sensitivity = UnityTargetSensitivity.HIGH
        requires_confirmation = True if policy is None else policy.require_confirmation_for_project_creation
        reason = "Unity project creation requires confirmation" if requires_confirmation else None
    elif operation_kind in {UnityEditorOperationKind.GENERATE_SCRIPT, UnityEditorOperationKind.WRITE_SCRIPT}:
        sensitivity = UnityTargetSensitivity.MEDIUM
        if would_overwrite and (policy is None or policy.require_confirmation_for_script_overwrite):
            sensitivity = UnityTargetSensitivity.HIGH
            requires_confirmation = True
            reason = "script overwrite requires confirmation"
            tags.append("overwrite")
    elif operation_kind == UnityEditorOperationKind.OPEN_PROJECT:
        sensitivity = UnityTargetSensitivity.MEDIUM
        requires_confirmation = True if policy is None else policy.require_confirmation_for_editor_open
        reason = "Unity editor open requires confirmation" if requires_confirmation else None
    elif operation_kind in {UnityEditorOperationKind.LAUNCH_PROJECT, UnityEditorOperationKind.OPEN_EDITOR}:
        sensitivity = UnityTargetSensitivity.HIGH
        requires_confirmation = True if policy is None else policy.require_confirmation_for_launch
        reason = "Unity editor launch requires confirmation" if requires_confirmation else None
    elif operation_kind == UnityEditorOperationKind.BRIDGE_COMMAND or uses_bridge:
        sensitivity = UnityTargetSensitivity.HIGH
        requires_confirmation = True if policy is None else policy.require_confirmation_for_bridge_commands
        reason = "Unity bridge command requires confirmation" if requires_confirmation else None
        tags.append("bridge")
    elif operation_kind in {
        UnityEditorOperationKind.OPEN_SCENE,
        UnityEditorOperationKind.LIST_LOADED_SCENES,
        UnityEditorOperationKind.REFRESH_ASSETS,
        UnityEditorOperationKind.CREATE_GAME_OBJECT,
        UnityEditorOperationKind.REQUEST_COMPILE,
        UnityEditorOperationKind.PING_BRIDGE,
        UnityEditorOperationKind.CONNECT_BRIDGE,
        UnityEditorOperationKind.DISCONNECT_BRIDGE,
    }:
        sensitivity = UnityTargetSensitivity.MEDIUM
        requires_confirmation = True if policy is None else policy.require_confirmation_for_editor_commands
        reason = "Unity editor command requires confirmation" if requires_confirmation else None

    if writes_project_settings:
        sensitivity = UnityTargetSensitivity.HIGH
        requires_confirmation = True
        reason = "ProjectSettings modifications require confirmation"
        tags.append("project_settings")

    if requires_confirmation:
        warnings.append(reason or "confirmation required")
    return UnityControlAdvice(
        requires_confirmation=requires_confirmation,
        reason=reason,
        sensitivity=sensitivity,
        policy_tags=tags,
        warnings=warnings,
        metadata={"project_root": project.project_root if project else None},
    )


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def validate_bridge_command(command_name: str, *, policy: UnitySafetyPolicy) -> None:
    lowered = command_name.strip().casefold()
    blocked = {item.casefold() for item in policy.blocked_bridge_commands}
    allowed = {item.casefold() for item in policy.allowed_bridge_commands}
    if lowered in blocked:
        raise UnitySafetyError(f"unity bridge command '{command_name}' is blocked")
    if allowed and lowered not in allowed:
        raise UnitySafetyError(f"unity bridge command '{command_name}' is not allowed")
