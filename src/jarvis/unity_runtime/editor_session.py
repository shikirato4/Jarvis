from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class UnityEditorSessionStatus(StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class UnityEditorSession(JarvisBaseModel):
    session_id: str
    project_root: str
    installation_id: str | None = None
    installation_path: str | None = None
    strategy: str | None = None
    transport_kind: str = "stub"
    endpoint: str | None = None
    status: UnityEditorSessionStatus = UnityEditorSessionStatus.DISCONNECTED
    connected: bool = False
    auto_connect: bool = False
    editor_pid: int | None = None
    launched_at: datetime | None = None
    connected_at: datetime | None = None
    last_health_at: datetime | None = None
    last_command_at: datetime | None = None
    last_error: str | None = None
    recent_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def disconnected(
        cls,
        *,
        session_id: str,
        project_root: str,
        transport_kind: str,
        installation_id: str | None = None,
        installation_path: str | None = None,
        strategy: str | None = None,
        endpoint: str | None = None,
        auto_connect: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> "UnityEditorSession":
        return cls(
            session_id=session_id,
            project_root=project_root,
            installation_id=installation_id,
            installation_path=installation_path,
            strategy=strategy,
            transport_kind=transport_kind,
            endpoint=endpoint,
            status=UnityEditorSessionStatus.DISCONNECTED,
            connected=False,
            auto_connect=auto_connect,
            metadata=metadata or {},
        )

    def mark_launched(self, *, editor_pid: int | None = None, launched_at: datetime | None = None) -> "UnityEditorSession":
        launched_at = launched_at or datetime.now(timezone.utc)
        return self.model_copy(
            update={
                "editor_pid": editor_pid,
                "launched_at": launched_at,
                "status": UnityEditorSessionStatus.CONNECTING if self.auto_connect else self.status,
            }
        )

    def mark_connected(
        self,
        *,
        endpoint: str | None = None,
        connected_at: datetime | None = None,
        last_health_at: datetime | None = None,
    ) -> "UnityEditorSession":
        connected_at = connected_at or datetime.now(timezone.utc)
        return self.model_copy(
            update={
                "endpoint": endpoint or self.endpoint,
                "connected": True,
                "status": UnityEditorSessionStatus.CONNECTED,
                "connected_at": connected_at,
                "last_health_at": last_health_at or connected_at,
                "last_error": None,
            }
        )

    def mark_disconnected(self, *, reason: str | None = None) -> "UnityEditorSession":
        errors = list(self.recent_errors)
        if reason:
            errors.append(reason)
        return self.model_copy(
            update={
                "connected": False,
                "status": UnityEditorSessionStatus.DISCONNECTED,
                "last_error": reason,
                "recent_errors": errors[-10:],
            }
        )

    def mark_degraded(self, *, reason: str | None = None) -> "UnityEditorSession":
        errors = list(self.recent_errors)
        if reason:
            errors.append(reason)
        return self.model_copy(
            update={
                "connected": False,
                "status": UnityEditorSessionStatus.DEGRADED,
                "last_error": reason,
                "recent_errors": errors[-10:],
                "last_health_at": datetime.now(timezone.utc),
            }
        )
