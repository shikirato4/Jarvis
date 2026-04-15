from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .editor_health import UnityBridgeHealth
from .editor_session import UnityEditorSession
from .launch import UnityLaunchResult


class UnityLaunchReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str = "unity.launch_project"
    success: bool
    message: str
    requires_confirmation: bool = False
    launch: UnityLaunchResult
    session: UnityEditorSession | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityBridgeOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    success: bool
    message: str
    command_name: str
    session: UnityEditorSession | None = None
    health: UnityBridgeHealth | None = None
    data: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
