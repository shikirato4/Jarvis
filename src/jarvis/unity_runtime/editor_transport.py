from __future__ import annotations

import json
import urllib.error
import urllib.request
from enum import StrEnum
from typing import Protocol

from jarvis.core.errors import UnityBridgeError

from .editor_commands import UnityBridgeCommand


class UnityBridgeTransportKind(StrEnum):
    HTTP_LOCAL = "http_local"
    SOCKET_LOCAL = "socket_local"
    NAMED_PIPE = "named_pipe"
    FILE_EXCHANGE = "file_exchange"
    STUB = "stub"


class UnityTransportResponse(dict):
    pass


class UnityEditorTransport(Protocol):
    transport_kind: UnityBridgeTransportKind

    def health(self, *, endpoint: str | None, timeout_ms: int) -> dict[str, object]: ...

    def send(
        self,
        *,
        endpoint: str | None,
        command_name: str,
        payload: dict[str, object],
        correlation_id: str,
        timeout_ms: int,
        metadata: dict[str, object],
    ) -> dict[str, object]: ...


class StubUnityEditorTransport:
    transport_kind = UnityBridgeTransportKind.STUB

    def health(self, *, endpoint: str | None, timeout_ms: int) -> dict[str, object]:
        return {
            "success": False,
            "status": "unavailable",
            "message": "stub transport is not connected",
            "endpoint": endpoint,
            "timeout_ms": timeout_ms,
        }

    def send(
        self,
        *,
        endpoint: str | None,
        command_name: str,
        payload: dict[str, object],
        correlation_id: str,
        timeout_ms: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        return {
            "success": False,
            "status": "unavailable",
            "message": "stub transport is not connected",
            "command_name": command_name,
            "endpoint": endpoint,
            "payload": payload,
            "correlation_id": correlation_id,
            "timeout_ms": timeout_ms,
            "metadata": metadata,
        }


class HttpLocalUnityEditorTransport:
    transport_kind = UnityBridgeTransportKind.HTTP_LOCAL

    def health(self, *, endpoint: str | None, timeout_ms: int) -> dict[str, object]:
        if not endpoint:
            return {
                "success": False,
                "status": "unavailable",
                "message": "missing unity bridge endpoint",
                "timeout_ms": timeout_ms,
            }
        return self.send(
            endpoint=endpoint,
            command_name=UnityBridgeCommand.PING.value,
            payload={},
            correlation_id="unity-bridge-health",
            timeout_ms=timeout_ms,
            metadata={},
        )

    def send(
        self,
        *,
        endpoint: str | None,
        command_name: str,
        payload: dict[str, object],
        correlation_id: str,
        timeout_ms: int,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if not endpoint:
            raise UnityBridgeError("unity bridge endpoint is not configured")
        request_payload = {
            "command": command_name,
            "payload": payload,
            "correlation_id": correlation_id,
            "metadata": metadata,
        }
        body = json.dumps(request_payload).encode("utf-8")
        request = urllib.request.Request(
            url=endpoint.rstrip("/") + "/jarvis/bridge",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(timeout_ms / 1000, 0.1)) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise UnityBridgeError(
                f"unity bridge request failed with status {exc.code}",
                details={"status_code": exc.code, "body": detail, "endpoint": endpoint, "command_name": command_name},
                recoverable=True,
            ) from exc
        except urllib.error.URLError as exc:
            raise UnityBridgeError(
                "unity bridge transport is unavailable",
                details={"endpoint": endpoint, "reason": str(exc.reason), "command_name": command_name},
                recoverable=True,
            ) from exc
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise UnityBridgeError(
                "unity bridge returned invalid json",
                details={"endpoint": endpoint, "command_name": command_name, "body": raw[:500]},
                recoverable=True,
            ) from exc
        if not isinstance(parsed, dict):
            raise UnityBridgeError(
                "unity bridge returned an invalid payload",
                details={"endpoint": endpoint, "command_name": command_name},
                recoverable=True,
            )
        return parsed
