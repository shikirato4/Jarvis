from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .base import UnityBridgeReceipt, UnityBridgeRequest, UnityBridgeStatus, UnityInstallationDescriptor
from .bridge import UnityBridgeBackend
from .editor_health import UnityBridgeHealth
from .editor_session import UnityEditorSession, UnityEditorSessionStatus
from .editor_transport import HttpLocalUnityEditorTransport, StubUnityEditorTransport, UnityBridgeTransportKind


class UnityInstallationProvider(Protocol):
    provider_name: str

    def list_installations(self) -> list[UnityInstallationDescriptor]: ...


class NativeUnityInstallationProvider:
    provider_name = "native_unity_installations"

    def __init__(self, *, known_locations: dict[str, str] | None = None, fallback_paths: tuple[str, ...] | None = None) -> None:
        self._known_locations = known_locations or {}
        self._fallback_paths = tuple(fallback_paths or ())

    def list_installations(self) -> list[UnityInstallationDescriptor]:
        candidates: list[Path] = []
        for value in self._known_locations.values():
            path = Path(value).expanduser()
            if path.exists():
                candidates.append(path)
        for raw in self._fallback_paths:
            path = Path(raw).expanduser()
            if path.exists():
                candidates.append(path)
        common = [
            Path("C:/Program Files/Unity/Hub/Editor"),
            Path("C:/Program Files/Unity"),
            Path.home() / "Unity",
            Path("/Applications/Unity"),
            Path("/Applications/Unity/Hub/Editor"),
        ]
        for path in common:
            if path.exists():
                candidates.append(path)
        installations: list[UnityInstallationDescriptor] = []
        seen: set[str] = set()
        for root in candidates:
            if root.is_file() and root.name.lower().startswith("unity"):
                editor = root
                version = self._guess_version(editor)
                if str(editor) in seen:
                    continue
                seen.add(str(editor))
                installations.append(
                    UnityInstallationDescriptor(
                        installation_id=str(editor),
                        version=version,
                        editor_path=str(editor),
                        hub_managed="Hub" in str(editor),
                        resolution_confidence=0.85,
                    )
                )
                continue
            for child in root.rglob("Unity.exe"):
                if str(child) in seen:
                    continue
                seen.add(str(child))
                installations.append(
                    UnityInstallationDescriptor(
                        installation_id=str(child),
                        version=self._guess_version(child),
                        editor_path=str(child),
                        hub_managed="Hub" in str(child),
                        resolution_confidence=0.9,
                    )
                )
            for child in root.rglob("Unity.app"):
                editor_path = child / "Contents/MacOS/Unity"
                if not editor_path.exists() or str(editor_path) in seen:
                    continue
                seen.add(str(editor_path))
                installations.append(
                    UnityInstallationDescriptor(
                        installation_id=str(editor_path),
                        version=self._guess_version(editor_path),
                        editor_path=str(editor_path),
                        hub_managed="Hub" in str(editor_path),
                        resolution_confidence=0.9,
                    )
                )
        return sorted(installations, key=lambda item: (item.version or "", item.editor_path), reverse=True)

    @staticmethod
    def _guess_version(editor_path: Path) -> str | None:
        parts = [part for part in editor_path.parts if any(ch.isdigit() for ch in part) and "." in part]
        return parts[-1] if parts else None


class InMemoryUnityInstallationProvider:
    provider_name = "in_memory_unity_installations"

    def __init__(self, installations: list[UnityInstallationDescriptor] | None = None) -> None:
        self._installations = installations or []

    def list_installations(self) -> list[UnityInstallationDescriptor]:
        return list(self._installations)


class NoOpUnityBridgeBackend:
    backend_name = "noop"

    def __init__(self) -> None:
        self._transport = StubUnityEditorTransport()

    def health(self, *, session: UnityEditorSession | None = None) -> UnityBridgeHealth:
        return UnityBridgeHealth(
            backend_name=self.backend_name,
            transport_kind=self._transport.transport_kind.value,
            status=UnityEditorSessionStatus.UNAVAILABLE,
            available=False,
            connected=False,
            session=session,
            warnings=["Unity bridge backend is not connected yet"],
            metadata=self._transport.health(endpoint=session.endpoint if session else None, timeout_ms=0),
        )

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
        session = UnityEditorSession.disconnected(
            session_id=str(uuid4()),
            project_root=project_root,
            installation_id=installation_id,
            installation_path=installation_path,
            strategy=strategy,
            transport_kind=self._transport.transport_kind.value,
            endpoint=endpoint,
            metadata=metadata,
        )
        return session.mark_degraded(reason="Unity bridge backend is not connected yet")

    def disconnect(self, session: UnityEditorSession | None) -> UnityEditorSession | None:
        if session is None:
            return None
        return session.mark_disconnected(reason="unity bridge disconnected")

    def send(self, request: UnityBridgeRequest, *, session: UnityEditorSession | None = None) -> UnityBridgeReceipt:
        return UnityBridgeReceipt(
            correlation_id=request.metadata.get("correlation_id", "unity-bridge-noop"),
            success=False,
            status=UnityBridgeStatus.PLANNED,
            message="Unity bridge backend is not connected yet",
            metadata=request.metadata,
            data={"command": request.command, "payload": request.payload, "session": session.model_dump(mode="json") if session else None},
        )


class HttpUnityBridgeBackend:
    backend_name = "http_local"

    def __init__(self, settings, *, logger=None, resilience_controller=None) -> None:
        self._settings = settings
        self._logger = logger
        self._transport = HttpLocalUnityEditorTransport()
        self._resilience = resilience_controller

    def health(self, *, session: UnityEditorSession | None = None) -> UnityBridgeHealth:
        endpoint = session.endpoint if session else self._default_endpoint()
        if self._resilience is not None:
            try:
                response, _ = self._resilience.execute(
                    service_name="unity_runtime",
                    dependency_name=self.backend_name,
                    operation_name="bridge.health",
                    timeout_ms=self._settings.unity_bridge_timeout_ms,
                    func=lambda: self._transport.health(endpoint=endpoint, timeout_ms=self._settings.unity_bridge_timeout_ms),
                )
            except Exception as exc:  # noqa: BLE001
                response = {"success": False, "message": str(exc)}
        else:
            response = self._transport.health(endpoint=endpoint, timeout_ms=self._settings.unity_bridge_timeout_ms)
        status = UnityEditorSessionStatus.CONNECTED if response.get("success") else UnityEditorSessionStatus.DEGRADED
        return UnityBridgeHealth(
            backend_name=self.backend_name,
            transport_kind=self._transport.transport_kind.value,
            status=status,
            available=True,
            connected=response.get("success", False),
            session=session,
            endpoint=(session.endpoint if session else None) or self._default_endpoint(),
            timeout_ms=self._settings.unity_bridge_timeout_ms,
            retry_count=self._settings.unity_bridge_retry_count,
            last_error=None if response.get("success") else response.get("message"),
            warnings=[] if response.get("success") else [str(response.get("message", "unity bridge unavailable"))],
            metadata=response,
        )

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
        session = UnityEditorSession.disconnected(
            session_id=str(uuid4()),
            project_root=project_root,
            installation_id=installation_id,
            installation_path=installation_path,
            strategy=strategy,
            transport_kind=self._transport.transport_kind.value,
            endpoint=endpoint or self._default_endpoint(),
            auto_connect=True,
            metadata=metadata,
        )
        response = self._transport.health(endpoint=session.endpoint, timeout_ms=self._settings.unity_bridge_timeout_ms)
        if response.get("success"):
            return session.mark_connected(endpoint=session.endpoint, connected_at=datetime.now(timezone.utc))
        return session.mark_degraded(reason=str(response.get("message", "unity bridge unavailable")))

    def disconnect(self, session: UnityEditorSession | None) -> UnityEditorSession | None:
        if session is None:
            return None
        return session.mark_disconnected(reason="unity bridge disconnected")

    def send(self, request: UnityBridgeRequest, *, session: UnityEditorSession | None = None) -> UnityBridgeReceipt:
        correlation_id = request.metadata.get("correlation_id", str(uuid4()))
        endpoint = request.metadata.get("endpoint") or (session.endpoint if session else None) or self._default_endpoint()
        timeout_ms = request.timeout_ms or self._settings.unity_bridge_timeout_ms
        warnings: list[str] = []
        response_payload: dict[str, object] | None = None
        last_error: str | None = None
        for _ in range(max(self._settings.unity_bridge_retry_count, 0) + 1):
            try:
                if self._resilience is not None:
                    response_payload, _ = self._resilience.execute(
                        service_name="unity_runtime",
                        dependency_name=self.backend_name,
                        operation_name=f"bridge.{request.command}",
                        timeout_ms=timeout_ms,
                        func=lambda: self._transport.send(
                            endpoint=endpoint,
                            command_name=request.command,
                            payload=request.payload,
                            correlation_id=correlation_id,
                            timeout_ms=timeout_ms,
                            metadata=request.metadata,
                        ),
                    )
                else:
                    response_payload = self._transport.send(
                        endpoint=endpoint,
                        command_name=request.command,
                        payload=request.payload,
                        correlation_id=correlation_id,
                        timeout_ms=timeout_ms,
                        metadata=request.metadata,
                    )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        if response_payload is None:
            return UnityBridgeReceipt(
                correlation_id=correlation_id,
                success=False,
                status=UnityBridgeStatus.DEGRADED,
                message=last_error or "unity bridge transport failed",
                metadata=request.metadata,
                warnings=warnings,
                data={"command": request.command, "payload": request.payload, "endpoint": endpoint},
            )
        success = bool(response_payload.get("success", False))
        status = UnityBridgeStatus.CONNECTED if success else UnityBridgeStatus.DEGRADED
        return UnityBridgeReceipt(
            correlation_id=str(response_payload.get("correlation_id", correlation_id)),
            success=success,
            status=status,
            message=str(response_payload.get("message", "unity bridge command executed")),
            metadata=request.metadata,
            warnings=list(response_payload.get("warnings", [])),
            data={
                "command": request.command,
                "payload": request.payload,
                "response": response_payload,
                "endpoint": endpoint,
                "transport_kind": self._transport.transport_kind.value,
            },
        )

    def _default_endpoint(self) -> str:
        return f"http://{self._settings.unity_bridge_host}:{self._settings.unity_bridge_port}"
