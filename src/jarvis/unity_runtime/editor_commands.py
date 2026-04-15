from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class UnityBridgeCommand(StrEnum):
    PING = "ping"
    OPEN_SCENE = "open_scene"
    LIST_SCENES = "list_scenes"
    LIST_LOADED_SCENES = "list_loaded_scenes"
    REVEAL_ASSET = "reveal_asset"
    REFRESH_ASSETS = "refresh_assets"
    CREATE_GAME_OBJECT = "create_gameobject"
    CAPTURE_SCENE_METADATA = "capture_scene_metadata"
    REQUEST_COMPILE = "request_compile"
    CUSTOM = "custom_command"


class UnityEditorCommandRequest(JarvisBaseModel):
    project: str
    command_name: UnityBridgeCommand | str
    payload: dict[str, object] = Field(default_factory=dict)
    timeout_ms: int | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class UnityEditorCommandResult(JarvisBaseModel):
    correlation_id: str
    command_name: str
    success: bool
    status: str
    message: str
    data: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

