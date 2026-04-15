from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class UnityProjectStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    BLOCKED = "blocked"


class UnityAssetKind(StrEnum):
    SCENE = "scene"
    SCRIPT = "script"
    PREFAB = "prefab"
    MATERIAL = "material"
    SHADER = "shader"
    SCRIPTABLE_OBJECT = "scriptable_object"
    ANIMATION = "animation"
    ASSEMBLY_DEFINITION = "assembly_definition"
    FOLDER = "folder"
    UNKNOWN = "unknown"


class UnityEditorOperationKind(StrEnum):
    OPEN_PROJECT = "open_project"
    LAUNCH_PROJECT = "launch_project"
    OPEN_EDITOR = "open_editor"
    CREATE_PROJECT = "create_project"
    LIST_SCENES = "list_scenes"
    LIST_ASSETS = "list_assets"
    GENERATE_SCRIPT = "generate_script"
    WRITE_SCRIPT = "write_script"
    OPEN_SCENE = "open_scene"
    LIST_LOADED_SCENES = "list_loaded_scenes"
    PING_BRIDGE = "ping_bridge"
    CONNECT_BRIDGE = "connect_bridge"
    DISCONNECT_BRIDGE = "disconnect_bridge"
    REFRESH_ASSETS = "refresh_assets"
    CREATE_GAME_OBJECT = "create_gameobject"
    REQUEST_COMPILE = "request_compile"
    REVEAL_ASSET = "reveal_asset"
    BRIDGE_COMMAND = "bridge_command"


class UnityLaunchStatus(StrEnum):
    OPENED = "opened"
    LAUNCHED = "launched"
    PREPARED = "prepared"
    CREATED = "created"
    WRITTEN = "written"
    BLOCKED = "blocked"
    CONFIRMATION_REQUIRED = "confirmation_required"
    FAILED = "failed"


class UnityBridgeStatus(StrEnum):
    UNAVAILABLE = "unavailable"
    PLANNED = "planned"
    CONNECTED = "connected"
    DEGRADED = "degraded"


class UnityTargetSensitivity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UnityInstallationDescriptor(JarvisBaseModel):
    installation_id: str
    version: str | None = None
    editor_path: str
    hub_managed: bool = False
    resolution_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnitySceneDescriptor(JarvisBaseModel):
    scene_name: str
    asset_path: str
    enabled_in_build: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityScriptDescriptor(JarvisBaseModel):
    class_name: str
    namespace: str | None = None
    asset_path: str
    folder_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityAssetDescriptor(JarvisBaseModel):
    asset_kind: UnityAssetKind
    name: str
    asset_path: str
    guid: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityProjectDescriptor(JarvisBaseModel):
    project_id: str
    name: str
    project_root: str
    assets_path: str
    packages_path: str
    project_settings_path: str
    unity_version: str | None = None
    status: UnityProjectStatus = UnityProjectStatus.RESOLVED
    is_valid_project: bool = False
    resolution_confidence: float = 0.0
    scenes: list[UnitySceneDescriptor] = Field(default_factory=list)
    detected_features: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityProjectQuery(JarvisBaseModel):
    query: str
    preferred_roots: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityProjectResolveRequest(JarvisBaseModel):
    query: UnityProjectQuery
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityProjectCreateRequest(JarvisBaseModel):
    name: str
    target_root: str
    template: str = "3d"
    unity_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityLaunchRequestModel(JarvisBaseModel):
    project: str
    installation_id: str | None = None
    strategy: str | None = None
    timeout_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityAssetSearchRequest(JarvisBaseModel):
    project: str
    query: str | None = None
    asset_kind: UnityAssetKind | None = None
    limit: int = 20
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityScriptGenerationRequest(JarvisBaseModel):
    project: str
    folder: str | None = None
    class_name: str
    namespace: str | None = None
    script_type: str = "mono_behaviour"
    overwrite: bool = False
    template_hints: dict[str, Any] = Field(default_factory=dict)
    base_class: str | None = None
    interfaces: list[str] = Field(default_factory=list)
    serialized_fields: list[dict[str, Any]] = Field(default_factory=list)
    using_directives: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityScriptWriteRequest(JarvisBaseModel):
    project: str
    asset_path: str | None = None
    folder: str | None = None
    class_name: str | None = None
    content: str
    overwrite: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityEditorOperationRequest(JarvisBaseModel):
    project: str
    operation_kind: UnityEditorOperationKind
    scene: str | None = None
    asset_path: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityBridgeRequest(JarvisBaseModel):
    project: str
    command: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityBridgeConnectRequest(JarvisBaseModel):
    project: str
    installation_id: str | None = None
    endpoint: str | None = None
    timeout_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityBridgeDisconnectRequest(JarvisBaseModel):
    project: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityControlAdvice(JarvisBaseModel):
    requires_confirmation: bool = False
    reason: str | None = None
    sensitivity: UnityTargetSensitivity = UnityTargetSensitivity.LOW
    policy_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UnityOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    success: bool
    status: UnityLaunchStatus
    message: str
    project: UnityProjectDescriptor | None = None
    scene: UnitySceneDescriptor | None = None
    asset: UnityAssetDescriptor | UnityScriptDescriptor | None = None
    written_paths: list[str] = Field(default_factory=list)
    confirmation_required: bool = False
    warnings: list[str] = Field(default_factory=list)
    would_overwrite: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityProjectResolveReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "unity.resolve_project"
    success: bool = True
    request: UnityProjectResolveRequest
    project: UnityProjectDescriptor
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityAssetSearchReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "unity.search_assets"
    success: bool = True
    request: UnityAssetSearchRequest
    project: UnityProjectDescriptor
    assets: list[UnityAssetDescriptor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityBridgeReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "unity.bridge"
    success: bool
    status: UnityBridgeStatus
    message: str
    project: UnityProjectDescriptor | None = None
    warnings: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
