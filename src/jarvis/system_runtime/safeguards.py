from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field

from jarvis.core.errors import SystemSafetyError
from jarvis.models.base import JarvisBaseModel

from .base import (
    ResolvedSystemTarget,
    SystemControlAdvice,
    SystemOpenMode,
    SystemOpenPolicy,
    SystemTargetKind,
    TargetSensitivity,
    VolumeDescriptor,
)
from .windows_apps import TRUSTED_WINDOWS_APP_IDS, TRUSTED_WINDOWS_EXECUTABLES, normalize_windows_app_name


class SystemSafetyPolicy(JarvisBaseModel):
    allowed_application_ids: list[str] = Field(default_factory=list)
    allowed_executables: list[str] = Field(default_factory=list)
    blocked_extensions: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    blocked_uri_schemes: list[str] = Field(default_factory=list)
    sensitive_roots: list[str] = Field(default_factory=list)
    require_confirmation_for_launch: bool = True
    require_confirmation_for_system_roots: bool = True
    require_confirmation_for_executable_open: bool = True
    require_confirmation_for_network_or_removable: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)


def build_system_open_policy(settings) -> SystemOpenPolicy:
    return SystemOpenPolicy(
        allow_direct_launch=True,
        allow_association_open=True,
        require_confirmation_for_sensitive_targets=True,
        blocked_extensions=[item.casefold() for item in settings.system_blocked_extensions],
        blocked_paths=[str(path) for path in settings.resolved_system_blocked_paths],
        blocked_apps=[],
        allowed_apps=[item.casefold() for item in settings.system_allowed_application_ids],
        allowed_uri_schemes=[item.casefold() for item in settings.system_allowed_uri_schemes],
        metadata={"backend_kind": settings.system_backend_kind},
    )


def build_system_safety_policy(settings) -> SystemSafetyPolicy:
    return SystemSafetyPolicy(
        allowed_application_ids=[item.casefold() for item in settings.system_allowed_application_ids],
        allowed_executables=[item.casefold() for item in settings.system_allowed_executables],
        blocked_extensions=[item.casefold() for item in settings.system_blocked_extensions],
        blocked_paths=[str(path) for path in settings.resolved_system_blocked_paths],
        blocked_uri_schemes=[item.casefold() for item in settings.system_blocked_uri_schemes],
        sensitive_roots=[str(path) for path in settings.resolved_system_sensitive_roots],
        require_confirmation_for_launch=settings.system_require_confirmation_for_launch,
        require_confirmation_for_system_roots=settings.system_require_confirmation_for_system_roots,
        require_confirmation_for_executable_open=settings.system_require_confirmation_for_executable_open,
        require_confirmation_for_network_or_removable=settings.system_require_confirmation_for_network_or_removable,
    )


def classify_target_sensitivity(
    target: ResolvedSystemTarget,
    *,
    policy: SystemSafetyPolicy,
    mode: SystemOpenMode,
    volume: VolumeDescriptor | None = None,
) -> tuple[TargetSensitivity, list[str]]:
    tags: list[str] = []
    sensitivity = TargetSensitivity.LOW

    if target.kind == SystemTargetKind.APPLICATION:
        sensitivity = TargetSensitivity.MEDIUM
        tags.append("kind:application")
    elif target.kind == SystemTargetKind.URI:
        sensitivity = TargetSensitivity.MEDIUM
        tags.append("kind:uri")

    path = Path(target.path).expanduser() if target.path else None
    if path is not None:
        suffix = path.suffix.casefold()
        if suffix in set(policy.blocked_extensions):
            sensitivity = TargetSensitivity.CRITICAL
            tags.append("blocked_extension")
        elif suffix in {".exe", ".com", ".msc", ".lnk"}:
            sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
            tags.append("executable")
        if _is_within_any(path, policy.sensitive_roots):
            sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
            tags.append("sensitive_root")
        if path.parent == path.anchor or str(path) == path.anchor:
            sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
            tags.append("volume_root")

    if target.kind == SystemTargetKind.URI and target.uri:
        scheme = urlparse(target.uri).scheme.casefold()
        if scheme in set(policy.blocked_uri_schemes):
            sensitivity = TargetSensitivity.CRITICAL
            tags.append("blocked_scheme")
        elif scheme not in {"http", "https", "mailto", "file"}:
            sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
            tags.append("custom_scheme")

    if mode == SystemOpenMode.LAUNCH_APPLICATION:
        sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
        tags.append("launch_mode")

    if volume is not None and (volume.is_network or volume.is_removable):
        sensitivity = max_sensitivity(sensitivity, TargetSensitivity.HIGH)
        tags.append("external_volume")

    return sensitivity, tags


def control_advice_for_target(
    target: ResolvedSystemTarget,
    *,
    policy: SystemSafetyPolicy,
    mode: SystemOpenMode,
    volume: VolumeDescriptor | None = None,
) -> SystemControlAdvice:
    warnings: list[str] = []
    sensitivity, tags = classify_target_sensitivity(target, policy=policy, mode=mode, volume=volume)
    requires_confirmation = False
    reason: str | None = None
    trusted_application = _is_trusted_application(target)

    path = Path(target.path).expanduser() if target.path else None
    if target.kind == SystemTargetKind.APPLICATION and policy.require_confirmation_for_launch and not trusted_application:
        requires_confirmation = True
        reason = "application launch requires confirmation"
    if path is not None and path.suffix.casefold() in set(policy.blocked_extensions):
        requires_confirmation = True
        reason = "target uses a blocked executable or script extension"
    if path is not None and _is_within_any(path, policy.sensitive_roots) and policy.require_confirmation_for_system_roots:
        requires_confirmation = True
        reason = "target is inside a sensitive root"
    if target.kind == SystemTargetKind.URI and target.uri:
        scheme = urlparse(target.uri).scheme.casefold()
        if scheme in set(policy.blocked_uri_schemes):
            raise SystemSafetyError(f"uri scheme '{scheme}' is blocked")
        if scheme not in {"http", "https", "mailto", "file"}:
            requires_confirmation = True
            reason = f"uri scheme '{scheme}' requires confirmation"
    if volume is not None and (volume.is_network or volume.is_removable) and policy.require_confirmation_for_network_or_removable:
        requires_confirmation = True
        reason = "target lives on a removable or network volume"
    if path is not None and _is_within_any(path, policy.blocked_paths):
        raise SystemSafetyError(f"path '{path}' is blocked by system policy")

    if requires_confirmation:
        warnings.append(reason or "confirmation required")
    return SystemControlAdvice(
        requires_confirmation=requires_confirmation,
        reason=reason,
        sensitivity=sensitivity,
        policy_tags=tags,
        warnings=warnings,
    )


def validate_allowed_target(target: ResolvedSystemTarget, *, policy: SystemSafetyPolicy, open_policy: SystemOpenPolicy) -> None:
    if target.kind == SystemTargetKind.APPLICATION:
        app_id = target.display_name.casefold()
        if open_policy.allowed_apps and app_id not in set(open_policy.allowed_apps):
            raise SystemSafetyError(f"application '{target.display_name}' is not in the allowed application list")
        executable_name = Path(target.path or "").name.casefold()
        if policy.allowed_executables and executable_name not in set(policy.allowed_executables):
            raise SystemSafetyError(f"executable '{executable_name}' is not in the allowed executable list")
    if target.kind in {SystemTargetKind.FILE, SystemTargetKind.FOLDER, SystemTargetKind.APPLICATION} and target.path:
        path = Path(target.path).expanduser()
        if _is_within_any(path, policy.blocked_paths):
            raise SystemSafetyError(f"path '{path}' is blocked by system policy")
        if path.suffix.casefold() in set(open_policy.blocked_extensions):
            raise SystemSafetyError(f"extension '{path.suffix}' is blocked by open policy")
    if target.kind == SystemTargetKind.URI and target.uri:
        scheme = urlparse(target.uri).scheme.casefold()
        if scheme not in set(open_policy.allowed_uri_schemes):
            raise SystemSafetyError(f"uri scheme '{scheme}' is not allowed")


def max_sensitivity(left: TargetSensitivity, right: TargetSensitivity) -> TargetSensitivity:
    order = {
        TargetSensitivity.LOW: 0,
        TargetSensitivity.MEDIUM: 1,
        TargetSensitivity.HIGH: 2,
        TargetSensitivity.CRITICAL: 3,
    }
    return left if order[left] >= order[right] else right


def _is_within_any(path: Path, roots: list[str]) -> bool:
    resolved = path.resolve(strict=False)
    for root in roots:
        candidate = Path(root).expanduser().resolve(strict=False)
        if resolved == candidate or resolved.is_relative_to(candidate):
            return True
    return False


def _is_trusted_application(target: ResolvedSystemTarget) -> bool:
    if target.kind != SystemTargetKind.APPLICATION:
        return False
    metadata = target.metadata or {}
    canonical_id = normalize_windows_app_name(str(metadata.get("canonical_app_id") or target.display_name))
    executable = Path(target.path or "").name.casefold()
    return canonical_id in TRUSTED_WINDOWS_APP_IDS or executable in TRUSTED_WINDOWS_EXECUTABLES
