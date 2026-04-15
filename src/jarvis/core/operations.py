from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterator
from uuid import uuid4

from jarvis.core.errors import JarvisError, ServiceUnavailableError
from jarvis.core.models import ActiveOperationRecord, AdmissionDecision, OperationStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class CancellationToken:
    operation_id: str
    correlation_id: str
    _event: threading.Event = field(default_factory=threading.Event)
    _reason: str | None = None

    def cancel(self, *, reason: str | None = None) -> None:
        self._reason = reason or self._reason or "cancelled"
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def raise_if_cancelled(self, *, component: str = "runtime") -> None:
        if self.cancelled():
            raise JarvisError(
                self._reason or "operation cancelled",
                code="operation_cancelled",
                component=component,
                details={"operation_id": self.operation_id, "correlation_id": self.correlation_id},
                recoverable=True,
            )


class OperationHandle:
    def __init__(self, registry: "OperationRegistry", record: ActiveOperationRecord, token: CancellationToken) -> None:
        self._registry = registry
        self.record = record
        self.token = token

    @property
    def operation_id(self) -> str:
        return self.record.operation_id

    @property
    def correlation_id(self) -> str:
        return self.record.correlation_id

    def heartbeat(self, *, progress_message: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        self._registry.heartbeat(self.operation_id, progress_message=progress_message, metadata=metadata)

    def set_status(self, status: OperationStatus, *, progress_message: str | None = None) -> None:
        self._registry.update_status(self.operation_id, status=status, progress_message=progress_message)

    def raise_if_cancelled(self, *, component: str | None = None) -> None:
        self.token.raise_if_cancelled(component=component or self.record.service_name)


class AdmissionController:
    def __init__(self, *, default_limit: int = 8, queue_limit: int = 16, policies: dict[str, dict[str, int | str]] | None = None) -> None:
        self._default_limit = max(default_limit, 1)
        self._default_queue_limit = max(queue_limit, 0)
        self._policies = policies or {}
        self._lock = threading.RLock()
        self._active: dict[str, set[str]] = {}
        self._queued: dict[str, int] = {}

    def admit(self, service_name: str, *, operation_id: str, overflow_policy: str | None = None) -> AdmissionDecision:
        with self._lock:
            policy = self._policies.get(service_name, {})
            limit = int(policy.get("max_concurrent", self._default_limit))
            queue_limit = int(policy.get("queue_limit", self._default_queue_limit))
            effective_overflow = str(policy.get("overflow_policy", overflow_policy or "reject"))
            active = self._active.setdefault(service_name, set())
            queued = self._queued.get(service_name, 0)
            if len(active) < limit:
                active.add(operation_id)
                return AdmissionDecision(service_name=service_name, granted=True, active_count=len(active), queue_depth=queued, limit=limit)
            if effective_overflow == "defer" and queued < queue_limit:
                self._queued[service_name] = queued + 1
                return AdmissionDecision(
                    service_name=service_name,
                    granted=False,
                    deferred=True,
                    reason="service queue saturated; operation deferred",
                    active_count=len(active),
                    queue_depth=self._queued[service_name],
                    limit=limit,
                )
            return AdmissionDecision(
                service_name=service_name,
                granted=False,
                reason="service concurrency limit reached",
                active_count=len(active),
                queue_depth=queued,
                limit=limit,
            )

    def release(self, service_name: str, operation_id: str) -> None:
        with self._lock:
            active = self._active.get(service_name)
            if active is not None:
                active.discard(operation_id)

    def activate_deferred(self, service_name: str, operation_id: str) -> bool:
        with self._lock:
            policy = self._policies.get(service_name, {})
            limit = int(policy.get("max_concurrent", self._default_limit))
            active = self._active.setdefault(service_name, set())
            queued = self._queued.get(service_name, 0)
            if len(active) >= limit or queued <= 0:
                return False
            self._queued[service_name] = max(queued - 1, 0)
            active.add(operation_id)
            return True

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            services = set(self._active) | set(self._queued) | set(self._policies)
            snapshot: list[dict[str, Any]] = []
            for service_name in sorted(services):
                policy = self._policies.get(service_name, {})
                snapshot.append(
                    {
                        "service_name": service_name,
                        "active_count": len(self._active.get(service_name, set())),
                        "queue_depth": self._queued.get(service_name, 0),
                        "limit": int(policy.get("max_concurrent", self._default_limit)),
                        "queue_limit": int(policy.get("queue_limit", self._default_queue_limit)),
                        "overflow_policy": str(policy.get("overflow_policy", "reject")),
                    }
                )
            return snapshot


class OperationRegistry:
    def __init__(self, *, event_bus=None, admission_controller: AdmissionController | None = None, history_limit: int = 200) -> None:
        self._event_bus = event_bus
        self._admission = admission_controller
        self._lock = threading.RLock()
        self._operations: dict[str, ActiveOperationRecord] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._callbacks: dict[str, Callable[[str], None]] = {}
        self._history: list[dict[str, Any]] = []
        self._history_limit = history_limit

    def begin(
        self,
        *,
        service_name: str,
        operation_name: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        watchdog_timeout_ms: int | None = None,
        timeout_hard: bool = False,
        overflow_policy: str | None = None,
        cancel_callback: Callable[[str], None] | None = None,
    ) -> OperationHandle:
        operation_id = str(uuid4())
        correlation = correlation_id or operation_id
        admission = self._admission.admit(service_name, operation_id=operation_id, overflow_policy=overflow_policy) if self._admission is not None else AdmissionDecision(service_name=service_name, granted=True)
        record = ActiveOperationRecord(
            operation_id=operation_id,
            correlation_id=correlation,
            service_name=service_name,
            operation_name=operation_name,
            status=OperationStatus.RUNNING if admission.granted else OperationStatus.DEFERRED if admission.deferred else OperationStatus.REJECTED,
            metadata=dict(metadata or {}),
            timeout_ms=timeout_ms,
            watchdog_timeout_ms=watchdog_timeout_ms or timeout_ms,
            timeout_hard=timeout_hard,
            admission=admission,
            deadline_at=_utcnow() + timedelta(milliseconds=timeout_ms) if timeout_ms is not None else None,
        )
        token = CancellationToken(operation_id=operation_id, correlation_id=correlation)
        with self._lock:
            self._operations[operation_id] = record
            self._tokens[operation_id] = token
            if cancel_callback is not None:
                self._callbacks[operation_id] = cancel_callback
            self._append_history({"operation_id": operation_id, "event": "started", "service_name": service_name, "operation_name": operation_name})
        self._publish("ops.operation.started", record)
        if not admission.granted:
            if admission.deferred:
                return OperationHandle(self, record, token)
            self.cancel(operation_id, reason=admission.reason or "admission rejected")
            raise ServiceUnavailableError(admission.reason or "operation admission rejected", details=admission.model_dump(mode="json"))
        return OperationHandle(self, record, token)

    @contextmanager
    def track(self, **kwargs: Any) -> Iterator[OperationHandle]:
        handle = self.begin(**kwargs)
        try:
            yield handle
        except Exception as exc:
            self.fail(handle.operation_id, error=str(exc))
            raise
        else:
            self.complete(handle.operation_id)

    def heartbeat(self, operation_id: str, *, progress_message: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            record = self._operations.get(operation_id)
            if record is None:
                return
            record.last_heartbeat_at = _utcnow()
            if progress_message is not None:
                record.progress_message = progress_message
            if metadata:
                record.metadata.update(metadata)

    def update_status(self, operation_id: str, *, status: OperationStatus, progress_message: str | None = None) -> None:
        with self._lock:
            record = self._operations.get(operation_id)
            if record is None:
                return
            record.status = status
            if progress_message is not None:
                record.progress_message = progress_message
            record.updated_at = _utcnow()

    def mark_running(self, operation_id: str) -> bool:
        with self._lock:
            record = self._operations.get(operation_id)
            if record is None:
                return False
            if self._admission is not None and record.status == OperationStatus.DEFERRED and not self._admission.activate_deferred(record.service_name, operation_id):
                return False
            record.status = OperationStatus.RUNNING
            record.updated_at = _utcnow()
            return True

    def complete(self, operation_id: str, *, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            record = self._operations.pop(operation_id, None)
            token = self._tokens.pop(operation_id, None)
            self._callbacks.pop(operation_id, None)
            if record is None:
                return
            if self._admission is not None:
                self._admission.release(record.service_name, operation_id)
            record.status = OperationStatus.COMPLETED if token is None or not token.cancelled() else OperationStatus.CANCELLED
            record.finished_at = _utcnow()
            if metadata:
                record.metadata.update(metadata)
            payload = record.model_dump(mode="json")
            self._append_history({"operation_id": operation_id, "event": "completed", "service_name": record.service_name, "operation_name": record.operation_name})
        self._publish("ops.operation.completed", payload)

    def fail(self, operation_id: str, *, error: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            record = self._operations.pop(operation_id, None)
            self._tokens.pop(operation_id, None)
            self._callbacks.pop(operation_id, None)
            if record is None:
                return
            if self._admission is not None:
                self._admission.release(record.service_name, operation_id)
            record.status = OperationStatus.FAILED
            record.last_error = error
            record.finished_at = _utcnow()
            if metadata:
                record.metadata.update(metadata)
            payload = record.model_dump(mode="json")
            self._append_history({"operation_id": operation_id, "event": "failed", "service_name": record.service_name, "operation_name": record.operation_name, "error": error})
        self._publish("ops.operation.failed", payload)

    def cancel(self, operation_id: str, *, reason: str = "cancel_requested") -> bool:
        callback = None
        with self._lock:
            record = self._operations.get(operation_id)
            token = self._tokens.get(operation_id)
            if record is None or token is None:
                return False
            if token.cancelled():
                return True
            record.status = OperationStatus.CANCELLING
            record.cancel_requested_at = _utcnow()
            record.progress_message = reason
            token.cancel(reason=reason)
            callback = self._callbacks.get(operation_id)
        if callback is not None:
            try:
                callback(reason)
            except Exception:
                pass
        self._publish("ops.operation.cancel_requested", {"operation_id": operation_id, "reason": reason})
        return True

    def cancel_by_correlation(self, correlation_id: str, *, reason: str = "cancel_requested") -> int:
        cancelled = 0
        for record in self.active_operations():
            if record.correlation_id == correlation_id:
                cancelled += 1 if self.cancel(record.operation_id, reason=reason) else 0
        return cancelled

    def cancel_by_metadata(self, key: str, value: Any, *, reason: str = "cancel_requested") -> int:
        cancelled = 0
        for record in self.active_operations():
            if record.metadata.get(key) == value:
                cancelled += 1 if self.cancel(record.operation_id, reason=reason) else 0
        return cancelled

    def token_for(self, operation_id: str) -> CancellationToken | None:
        with self._lock:
            return self._tokens.get(operation_id)

    def active_operations(self) -> list[ActiveOperationRecord]:
        with self._lock:
            return [record.model_copy(deep=True) for record in self._operations.values()]

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def snapshot(self) -> dict[str, Any]:
        operations = self.active_operations()
        return {
            "active_count": len(operations),
            "operations": [record.model_dump(mode="json") for record in operations],
            "queues": self._admission.snapshot() if self._admission is not None else [],
        }

    def _append_history(self, item: dict[str, Any]) -> None:
        item["recorded_at"] = _utcnow()
        self._history.insert(0, item)
        if len(self._history) > self._history_limit:
            del self._history[self._history_limit :]

    def _publish(self, event_name: str, payload: ActiveOperationRecord | dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        if isinstance(payload, ActiveOperationRecord):
            self._event_bus.publish(event_name, payload.model_dump(mode="json"))
        else:
            self._event_bus.publish(event_name, payload)


class OperationWatchdog:
    def __init__(
        self,
        registry: OperationRegistry,
        *,
        event_bus=None,
        poll_interval_seconds: float = 1.0,
        hard_timeout_callback: Callable[[ActiveOperationRecord], None] | None = None,
    ) -> None:
        self._registry = registry
        self._event_bus = event_bus
        self._poll_interval_seconds = max(poll_interval_seconds, 0.2)
        self._hard_timeout_callback = hard_timeout_callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-operation-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_seconds):
            now = _utcnow()
            for record in self._registry.active_operations():
                if record.status not in {OperationStatus.RUNNING, OperationStatus.CANCELLING, OperationStatus.DEFERRED}:
                    continue
                heartbeat_age_ms = (now - record.last_heartbeat_at).total_seconds() * 1000
                timeout_ms = record.watchdog_timeout_ms or record.timeout_ms
                if timeout_ms is not None and heartbeat_age_ms > timeout_ms:
                    self._publish_timeout(record, heartbeat_age_ms)
                    self._registry.cancel(record.operation_id, reason="watchdog timeout")
                    if record.timeout_hard and self._hard_timeout_callback is not None:
                        self._hard_timeout_callback(record)
                elif record.deadline_at is not None and now > record.deadline_at:
                    self._publish_deadline(record)
                    self._registry.cancel(record.operation_id, reason="operation deadline exceeded")

    def _publish_timeout(self, record: ActiveOperationRecord, heartbeat_age_ms: float) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                "ops.watchdog.timeout",
                {
                    "operation_id": record.operation_id,
                    "service_name": record.service_name,
                    "operation_name": record.operation_name,
                    "heartbeat_age_ms": heartbeat_age_ms,
                },
            )

    def _publish_deadline(self, record: ActiveOperationRecord) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                "ops.watchdog.deadline_exceeded",
                {
                    "operation_id": record.operation_id,
                    "service_name": record.service_name,
                    "operation_name": record.operation_name,
                    "deadline_at": record.deadline_at.isoformat() if record.deadline_at else None,
                },
            )
