from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .base import UnityBridgeReceipt, UnityBridgeRequest, UnityBridgeStatus
from .editor_health import UnityBridgeHealth
from .editor_session import UnityEditorSession, UnityEditorSessionStatus


class UnityBridgeResponse(JarvisBaseModel):
    correlation_id: str
    success: bool
    status: str
    command_name: str
    message: str
    data: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UnityBridgeBackend(Protocol):
    backend_name: str

    def health(self, *, session: UnityEditorSession | None = None) -> UnityBridgeHealth: ...

    def connect(
        self,
        *,
        project_root: str,
        endpoint: str | None = None,
        installation_id: str | None = None,
        installation_path: str | None = None,
        strategy: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UnityEditorSession: ...

    def disconnect(self, session: UnityEditorSession | None) -> UnityEditorSession | None: ...

    def send(self, request: UnityBridgeRequest, *, session: UnityEditorSession | None = None) -> UnityBridgeReceipt: ...


class UnityBridgeService:
    def __init__(self, registry, *, backend_name: str, session_registry=None, logger=None) -> None:
        self._registry = registry
        self._backend_name = backend_name
        self._sessions = session_registry
        self._logger = logger

    def backend(self):
        return self._registry.get(self._backend_name)

    def health(self, *, project_root: str | None = None) -> dict[str, object]:
        backend = self.backend()
        session = self._sessions.get(project_root) if project_root and self._sessions else None
        if backend is None:
            return UnityBridgeHealth(
                backend_name=self._backend_name,
                transport_kind="stub",
                status=UnityEditorSessionStatus.UNAVAILABLE,
                available=False,
                connected=False,
                session=session,
                warnings=[f"unity bridge backend '{self._backend_name}' is unavailable"],
            ).model_dump(mode="json")
        return backend.health(session=session).model_dump(mode="json")

    def connect(
        self,
        *,
        project_root: str,
        endpoint: str | None = None,
        installation_id: str | None = None,
        installation_path: str | None = None,
        strategy: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UnityEditorSession:
        backend = self.backend()
        if backend is None:
            session = UnityEditorSession.disconnected(
                session_id=str(uuid4()),
                project_root=project_root,
                transport_kind="stub",
                installation_id=installation_id,
                installation_path=installation_path,
                strategy=strategy,
                endpoint=endpoint,
                metadata=metadata,
            ).mark_degraded(reason=f"unity bridge backend '{self._backend_name}' is unavailable")
            if self._sessions is not None:
                self._sessions.save(session)
            return session
        session = backend.connect(
            project_root=project_root,
            endpoint=endpoint,
            installation_id=installation_id,
            installation_path=installation_path,
            strategy=strategy,
            metadata=metadata,
        )
        if self._sessions is not None:
            self._sessions.save(session)
        return session

    def disconnect(self, *, project_root: str) -> UnityEditorSession | None:
        session = self._sessions.get(project_root) if self._sessions is not None else None
        backend = self.backend()
        if backend is None:
            if session is not None and self._sessions is not None:
                session = session.mark_disconnected(reason="unity bridge backend unavailable")
                self._sessions.save(session)
            return session
        disconnected = backend.disconnect(session)
        if disconnected is not None and self._sessions is not None:
            self._sessions.save(disconnected)
        elif self._sessions is not None and session is not None:
            self._sessions.remove(project_root)
        return disconnected

    def send(self, request: UnityBridgeRequest) -> UnityBridgeReceipt:
        backend = self.backend()
        session = self._sessions.get(request.project) if self._sessions is not None else None
        if backend is None:
            return UnityBridgeReceipt(
                correlation_id=request.metadata.get("correlation_id", "unity-bridge-unavailable"),
                success=False,
                status=UnityBridgeStatus.UNAVAILABLE,
                message=f"unity bridge backend '{self._backend_name}' is unavailable",
                metadata=request.metadata,
                data={"command": request.command, "payload": request.payload},
            )
        receipt = backend.send(request, session=session)
        updated = None
        if receipt.success and session is not None:
            updated = session.mark_connected(endpoint=session.endpoint)
        elif session is not None and not receipt.success:
            updated = session.mark_degraded(reason=receipt.message)
        if updated is not None and self._sessions is not None:
            self._sessions.save(updated)
        return receipt
