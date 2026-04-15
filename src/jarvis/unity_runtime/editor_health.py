from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .editor_session import UnityEditorSession, UnityEditorSessionStatus


class UnityBridgeHealth(JarvisBaseModel):
    backend_name: str
    transport_kind: str
    status: UnityEditorSessionStatus = UnityEditorSessionStatus.UNAVAILABLE
    available: bool = False
    connected: bool = False
    session: UnityEditorSession | None = None
    endpoint: str | None = None
    timeout_ms: int | None = None
    retry_count: int = 0
    last_checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

