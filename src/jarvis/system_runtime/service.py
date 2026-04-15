from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from jarvis.core.errors import SystemLaunchError, SystemResolutionError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .base import (
    ResolvedSystemTarget,
    SystemLaunchStatus,
    SystemOpenMode,
    SystemOpenRequest,
    SystemOperationReceipt,
    SystemResolveReceipt,
    SystemResolveRequest,
    SystemSearchReceipt,
    SystemSearchRequest,
    SystemTargetKind,
)
from .safeguards import build_system_open_policy, build_system_safety_policy, control_advice_for_target, validate_allowed_target


class SystemRuntimeService(RuntimeServiceContract):
    service_name = "system_runtime"

    def __init__(
        self,
        settings,
        event_bus,
        topology,
        search_service,
        resolver,
        launcher,
        *,
        logger: logging.Logger | None = None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._event_bus = event_bus
        self._topology = topology
        self._search = search_service
        self._resolver = resolver
        self._launcher = launcher
        self._logger = logger or logging.getLogger("jarvis.system")
        self._operations = operation_registry
        self._started = False
        self._open_policy = build_system_open_policy(settings)
        self._safety_policy = build_system_safety_policy(settings)

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.service_name,
            status=HealthStatus.READY if self._started and self._settings.system_runtime_enabled else HealthStatus.STOPPED,
            details=self.status(),
        )

    def status(self) -> dict[str, object]:
        return {
            "enabled": self._settings.system_runtime_enabled,
            "backend_kind": self._settings.system_backend_kind,
            "search_roots": [str(path) for path in self._topology.default_search_roots()],
            "volumes": [item.model_dump(mode="json") for item in self._topology.list_volumes()],
            "known_locations": [item.model_dump(mode="json") for item in self._topology.list_known_locations()],
            "blocked_extensions": list(self._settings.system_blocked_extensions),
            "allowed_uri_schemes": list(self._settings.system_allowed_uri_schemes),
        }

    def search(self, request: SystemSearchRequest | dict) -> SystemSearchReceipt:
        self._ensure_started()
        payload = SystemSearchRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        handle = self._begin_operation("system.search", payload.metadata, timeout_ms=self._settings.system_operation_timeout_ms)
        try:
            matches = self._search.search(payload.resource)
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"match_count": len(matches)})
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc))
            raise
        receipt = SystemSearchReceipt(
            correlation_id=str(uuid4()),
            query=payload.resource,
            matches=matches,
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("system.search", receipt.model_dump(mode="json"))
        return receipt

    def resolve(self, request: SystemResolveRequest | dict) -> SystemResolveReceipt:
        self._ensure_started()
        payload = SystemResolveRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        handle = self._begin_operation("system.resolve", payload.metadata, timeout_ms=self._settings.system_operation_timeout_ms)
        try:
            target = self._resolver.resolve(payload)
            target = self._harden_resolved_target(target)
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"target_kind": target.kind.value})
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc))
            raise
        target = self._decorate_target(target, mode=SystemOpenMode.DEFAULT)
        receipt = SystemResolveReceipt(
            correlation_id=str(uuid4()),
            request=payload,
            resolved_target=target,
            warnings=target.warnings,
            metadata=payload.metadata,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("system.resolve", receipt.model_dump(mode="json"))
        return receipt

    def open(self, request: SystemOpenRequest | dict, *, correlation_id: str | None = None) -> SystemOperationReceipt:
        self._ensure_started()
        payload = SystemOpenRequest.model_validate(request)
        started_at = datetime.now(timezone.utc)
        handle = self._begin_operation("system.open", payload.metadata, timeout_ms=self._settings.system_operation_timeout_ms, correlation_id=correlation_id)
        target = self._resolve_open_target(payload)
        target = self._harden_resolved_target(target)
        target = self._decorate_target(target, mode=self._effective_mode(payload, target))
        advice = control_advice_for_target(
            target,
            policy=self._safety_policy,
            mode=self._effective_mode(payload, target),
            volume=self._find_volume_for_target(target),
        )
        target.sensitivity = advice.sensitivity
        target.warnings.extend(item for item in advice.warnings if item not in target.warnings)
        validate_allowed_target(target, policy=self._safety_policy, open_policy=self._open_policy)
        if advice.requires_confirmation and not payload.metadata.get("approved", False):
            receipt = SystemOperationReceipt(
                correlation_id=correlation_id or str(uuid4()),
                operation_name="system.open",
                success=False,
                status=SystemLaunchStatus.CONFIRMATION_REQUIRED,
                message=advice.reason or "confirmation required",
                resolved_target=target,
                association=target.association,
                confirmation_required=True,
                sensitivity=advice.sensitivity,
                warnings=target.warnings,
                data={"policy_tags": advice.policy_tags},
                metadata=payload.metadata,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._event_bus.publish("system.confirmation_required", receipt.model_dump(mode="json"))
            if handle is not None:
                self._operations.complete(handle.operation_id, metadata={"status": receipt.status.value})
            return receipt
        try:
            launch_data = self._launcher.launch(target, mode=self._effective_mode(payload, target), dry_run=payload.dry_run)
        except Exception as exc:
            if handle is not None:
                self._operations.fail(handle.operation_id, error=str(exc))
            raise
        status = SystemLaunchStatus.LAUNCHED if target.kind == SystemTargetKind.APPLICATION else SystemLaunchStatus.OPENED
        if self._effective_mode(payload, target) == SystemOpenMode.REVEAL_IN_FOLDER:
            status = SystemLaunchStatus.OPENED
        receipt = SystemOperationReceipt(
            correlation_id=correlation_id or str(uuid4()),
            operation_name="system.open",
            success=True,
            status=status,
            message="system target opened",
            resolved_target=target,
            association=target.association,
            confirmation_required=False,
            sensitivity=advice.sensitivity,
            warnings=target.warnings,
            data=launch_data,
            metadata=payload.metadata | {"policy_tags": advice.policy_tags},
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._event_bus.publish("system.opened", receipt.model_dump(mode="json"))
        if handle is not None:
            self._operations.complete(handle.operation_id, metadata={"status": receipt.status.value})
        return receipt

    def open_path(self, path: str, *, reveal_in_folder: bool = False, dry_run: bool = False, metadata: dict | None = None):
        return self.open(SystemOpenRequest(path=path, reveal_in_folder=reveal_in_folder, dry_run=dry_run, metadata=metadata or {}))

    def open_application(self, application: str, *, dry_run: bool = False, metadata: dict | None = None):
        return self.open(
            SystemOpenRequest(application=application, mode=SystemOpenMode.LAUNCH_APPLICATION, dry_run=dry_run, metadata=metadata or {})
        )

    def reveal(self, path: str, *, dry_run: bool = False, metadata: dict | None = None):
        return self.open(
            SystemOpenRequest(path=path, mode=SystemOpenMode.REVEAL_IN_FOLDER, reveal_in_folder=True, dry_run=dry_run, metadata=metadata or {})
        )

    def _resolve_open_target(self, request: SystemOpenRequest) -> ResolvedSystemTarget:
        if request.path:
            target = self._resolver.resolve(SystemResolveRequest(query=request.path, metadata=request.metadata))
        elif request.uri:
            target = self._resolver.resolve(SystemResolveRequest(query=request.uri, target_kind=SystemTargetKind.URI, metadata=request.metadata))
        elif request.application:
            target = self._resolver.resolve(
                SystemResolveRequest(query=request.application, target_kind=SystemTargetKind.APPLICATION, search_scope="applications", metadata=request.metadata)
            )
        elif request.query:
            target = self._resolver.resolve(SystemResolveRequest(query=request.query, metadata=request.metadata))
        else:
            raise SystemResolutionError("system open request requires query, path, uri or application")
        return self._resolver.require_resolved(target)

    def _decorate_target(self, target: ResolvedSystemTarget, *, mode: SystemOpenMode) -> ResolvedSystemTarget:
        volume = self._find_volume_for_target(target)
        advice = control_advice_for_target(target, policy=self._safety_policy, mode=mode, volume=volume)
        target.sensitivity = advice.sensitivity
        target.warnings = list(dict.fromkeys([*target.warnings, *advice.warnings]))
        return target

    def _harden_resolved_target(self, target: ResolvedSystemTarget) -> ResolvedSystemTarget:
        if target.path is None:
            return target
        normalized = Path(target.path).expanduser().resolve(strict=False)
        if len(str(normalized)) > self._settings.system_max_path_length:
            raise SystemResolutionError("system target path exceeds maximum length", details={"path": str(normalized)})
        if ".." in normalized.parts:
            raise SystemResolutionError("system target path is not canonical", details={"path": str(normalized)})
        target.path = str(normalized)
        return target

    def _begin_operation(self, operation_name: str, metadata: dict, *, timeout_ms: int, correlation_id: str | None = None):
        if self._operations is None:
            return None
        return self._operations.begin(
            service_name=self.service_name,
            operation_name=operation_name,
            correlation_id=correlation_id,
            metadata=metadata,
            timeout_ms=timeout_ms,
            watchdog_timeout_ms=timeout_ms,
        )

    def _find_volume_for_target(self, target: ResolvedSystemTarget):
        if target.path is None:
            return None
        path = Path(target.path).resolve(strict=False)
        for volume in self._topology.list_volumes():
            root = Path(volume.root).resolve(strict=False)
            if path == root or path.is_relative_to(root):
                return volume
        return None

    @staticmethod
    def _effective_mode(request: SystemOpenRequest, target: ResolvedSystemTarget) -> SystemOpenMode:
        if request.reveal_in_folder or request.mode == SystemOpenMode.REVEAL_IN_FOLDER:
            return SystemOpenMode.REVEAL_IN_FOLDER
        if request.mode != SystemOpenMode.DEFAULT:
            return request.mode
        if target.kind == SystemTargetKind.APPLICATION:
            return SystemOpenMode.LAUNCH_APPLICATION
        return SystemOpenMode.OPEN_WITH_ASSOCIATION

    def _ensure_started(self) -> None:
        if not self._started or not self._settings.system_runtime_enabled:
            raise SystemLaunchError("system runtime is not started or disabled")
