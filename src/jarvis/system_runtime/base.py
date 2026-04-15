from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class SystemTargetKind(StrEnum):
    APPLICATION = "application"
    FILE = "file"
    FOLDER = "folder"
    URI = "uri"
    UNKNOWN = "unknown"


class SystemSearchScope(StrEnum):
    CONFIGURED_ROOTS = "configured_roots"
    MOUNTED_VOLUMES = "mounted_volumes"
    KNOWN_LOCATIONS = "known_locations"
    APPLICATIONS = "applications"
    ALL = "all"


class SystemOpenMode(StrEnum):
    DEFAULT = "default"
    OPEN_WITH_ASSOCIATION = "open_with_association"
    LAUNCH_APPLICATION = "launch_application"
    REVEAL_IN_FOLDER = "reveal_in_folder"


class TargetSensitivity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    BLOCKED = "blocked"


class SystemLaunchStatus(StrEnum):
    OPENED = "opened"
    LAUNCHED = "launched"
    BLOCKED = "blocked"
    CONFIRMATION_REQUIRED = "confirmation_required"
    FAILED = "failed"


class VolumeDescriptor(JarvisBaseModel):
    name: str
    root: str
    kind: str = "filesystem"
    is_ready: bool = True
    is_removable: bool = False
    is_network: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnownLocationDescriptor(JarvisBaseModel):
    location_id: str
    label: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceQuery(JarvisBaseModel):
    query: str
    target_kind: SystemTargetKind | None = None
    search_scope: SystemSearchScope = SystemSearchScope.ALL
    preferred_roots: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    max_results: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceMatch(JarvisBaseModel):
    match_id: str
    display_name: str
    kind: SystemTargetKind
    path: str | None = None
    score: float = 0.0
    volume_root: str | None = None
    exists: bool = True
    resolution_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssociationResolution(JarvisBaseModel):
    target_kind: SystemTargetKind
    handler_kind: str
    handler_name: str | None = None
    handler_path: str | None = None
    supports_open: bool = True
    supports_reveal: bool = False
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResolvedSystemTarget(JarvisBaseModel):
    target_id: str
    kind: SystemTargetKind
    display_name: str
    path: str | None = None
    uri: str | None = None
    command: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    association: AssociationResolution | None = None
    sensitivity: TargetSensitivity = TargetSensitivity.LOW
    resolution_status: SystemResolutionStatus = SystemResolutionStatus.RESOLVED
    ambiguity_candidates: list[ResourceMatch] = Field(default_factory=list)
    resolution_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemOpenPolicy(JarvisBaseModel):
    allow_direct_launch: bool = True
    allow_association_open: bool = True
    require_confirmation_for_sensitive_targets: bool = True
    blocked_extensions: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    blocked_apps: list[str] = Field(default_factory=list)
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_uri_schemes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemSearchRequest(JarvisBaseModel):
    resource: ResourceQuery
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemResolveRequest(JarvisBaseModel):
    query: str
    target_kind: SystemTargetKind | None = None
    search_scope: SystemSearchScope = SystemSearchScope.ALL
    preferred_roots: list[str] = Field(default_factory=list)
    extensions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemOpenRequest(JarvisBaseModel):
    query: str | None = None
    path: str | None = None
    uri: str | None = None
    application: str | None = None
    target_id: str | None = None
    mode: SystemOpenMode = SystemOpenMode.DEFAULT
    reveal_in_folder: bool = False
    dry_run: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemControlAdvice(JarvisBaseModel):
    requires_confirmation: bool = False
    reason: str | None = None
    sensitivity: TargetSensitivity = TargetSensitivity.LOW
    policy_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    success: bool
    status: SystemLaunchStatus
    message: str
    resolved_target: ResolvedSystemTarget | None = None
    association: AssociationResolution | None = None
    confirmation_required: bool = False
    sensitivity: TargetSensitivity = TargetSensitivity.LOW
    warnings: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemSearchReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "system.search"
    success: bool = True
    query: ResourceQuery
    matches: list[ResourceMatch] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemResolveReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "system.resolve"
    success: bool = True
    request: SystemResolveRequest
    resolved_target: ResolvedSystemTarget
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
