from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .models import AutoRecoveryPolicy, RecoveryRecord, ServiceLifecycleRecord, ServiceLifecycleState
from .telemetry import TelemetryRecorder


class RecoveryPlan(JarvisBaseModel):
    service_name: str
    dry_run: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


class RecoveryResult(JarvisBaseModel):
    service_name: str
    success: bool
    message: str
    record: ServiceLifecycleRecord
    recovery: RecoveryRecord | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RuntimeLifecycleSupervisor:
    def __init__(self, telemetry: TelemetryRecorder, *, event_bus=None) -> None:
        self._telemetry = telemetry
        self._event_bus = event_bus
        self._services: dict[str, dict[str, Callable[[], None] | None]] = {}
        self._records: dict[str, ServiceLifecycleRecord] = {}
        self._auto_recovery: dict[str, AutoRecoveryPolicy] = {}
        self._recent_recoveries: dict[str, deque[datetime]] = defaultdict(deque)
        self._last_auto_recovery_at: dict[str, datetime] = {}

    def register(self, name: str, *, start: Callable[[], None], stop: Callable[[], None], health: Callable[[], object] | None = None) -> None:
        self._services[name] = {"start": start, "stop": stop, "health": health}
        self._records.setdefault(name, ServiceLifecycleRecord(service_name=name, state=ServiceLifecycleState.CREATED))

    def configure_auto_recovery(self, service_name: str, policy: AutoRecoveryPolicy) -> None:
        self._auto_recovery[service_name] = policy

    def start_service(self, service_name: str) -> ServiceLifecycleRecord:
        managed = self._services[service_name]
        current = self._records.get(service_name) or ServiceLifecycleRecord(service_name=service_name, state=ServiceLifecycleState.CREATED)
        self._records[service_name] = current.model_copy(update={"state": ServiceLifecycleState.STARTING, "start_count": current.start_count + 1})
        started = time.perf_counter()
        try:
            managed["start"]()
            record = self._records[service_name].model_copy(update={"state": ServiceLifecycleState.READY, "latency_ms": (time.perf_counter() - started) * 1000})
            self._records[service_name] = record
            self._publish(service_name, record.state)
            return record
        except Exception as exc:  # noqa: BLE001
            record = self._records[service_name].model_copy(update={"state": ServiceLifecycleState.FAILED, "last_error": str(exc), "latency_ms": (time.perf_counter() - started) * 1000})
            self._records[service_name] = record
            self._telemetry.record_failure(service_name=service_name, operation_name="start", error=str(exc))
            self._publish(service_name, record.state, error=str(exc))
            raise

    def stop_service(self, service_name: str) -> ServiceLifecycleRecord:
        managed = self._services[service_name]
        current = self._records.get(service_name) or ServiceLifecycleRecord(service_name=service_name, state=ServiceLifecycleState.CREATED)
        self._records[service_name] = current.model_copy(update={"state": ServiceLifecycleState.STOPPING, "stop_count": current.stop_count + 1})
        started = time.perf_counter()
        try:
            managed["stop"]()
            record = self._records[service_name].model_copy(update={"state": ServiceLifecycleState.STOPPED, "latency_ms": (time.perf_counter() - started) * 1000})
            self._records[service_name] = record
            self._publish(service_name, record.state)
            return record
        except Exception as exc:  # noqa: BLE001
            record = self._records[service_name].model_copy(update={"state": ServiceLifecycleState.FAILED, "last_error": str(exc), "latency_ms": (time.perf_counter() - started) * 1000})
            self._records[service_name] = record
            self._telemetry.record_failure(service_name=service_name, operation_name="stop", error=str(exc))
            self._publish(service_name, record.state, error=str(exc))
            raise

    def recover_service(self, plan: RecoveryPlan) -> RecoveryResult:
        record = self._records.get(plan.service_name) or ServiceLifecycleRecord(service_name=plan.service_name, state=ServiceLifecycleState.CREATED)
        if plan.dry_run:
            return RecoveryResult(service_name=plan.service_name, success=True, message="dry-run recovery plan generated", record=record, metadata=plan.metadata)
        self._records[plan.service_name] = record.model_copy(update={"state": ServiceLifecycleState.RECOVERING, "recover_count": record.recover_count + 1})
        self._publish(plan.service_name, ServiceLifecycleState.RECOVERING)
        try:
            self.stop_service(plan.service_name)
        except Exception:
            pass
        try:
            next_record = self.start_service(plan.service_name)
            recovery = self._telemetry.record_recovery(service_name=plan.service_name, success=True, message="service recovered", metadata=plan.metadata)
            if self._event_bus is not None:
                self._event_bus.publish("ops.recovery_recorded", recovery.model_dump(mode="json"))
            return RecoveryResult(service_name=plan.service_name, success=True, message="service recovered", record=next_record, recovery=recovery, metadata=plan.metadata)
        except Exception as exc:  # noqa: BLE001
            failed = self._records[plan.service_name].model_copy(update={"state": ServiceLifecycleState.FAILED, "last_error": str(exc)})
            self._records[plan.service_name] = failed
            recovery = self._telemetry.record_recovery(service_name=plan.service_name, success=False, message=str(exc), metadata=plan.metadata)
            if self._event_bus is not None:
                self._event_bus.publish("ops.recovery_recorded", recovery.model_dump(mode="json"))
            self._publish(plan.service_name, ServiceLifecycleState.FAILED, error=str(exc))
            return RecoveryResult(service_name=plan.service_name, success=False, message=str(exc), record=failed, recovery=recovery, metadata=plan.metadata)

    def attempt_auto_recover(self, service_name: str, *, reason: str, metadata: dict[str, object] | None = None) -> RecoveryResult | None:
        policy = self._auto_recovery.get(service_name)
        if policy is None or not policy.enabled:
            return None
        now = datetime.now(timezone.utc)
        last_attempt = self._last_auto_recovery_at.get(service_name)
        if last_attempt is not None and (now - last_attempt).total_seconds() < policy.cooldown_seconds:
            return None
        window = self._recent_recoveries[service_name]
        while window and (now - window[-1]).total_seconds() > policy.window_seconds:
            window.pop()
        if len(window) >= policy.max_attempts_per_window:
            return None
        self._last_auto_recovery_at[service_name] = now
        window.appendleft(now)
        result = self.recover_service(RecoveryPlan(service_name=service_name, metadata={"trigger": "auto_recover", "reason": reason, **(metadata or {})}))
        return result

    def records(self) -> list[ServiceLifecycleRecord]:
        return list(self._records.values())

    def _publish(self, service_name: str, state: ServiceLifecycleState, *, error: str | None = None) -> None:
        if self._event_bus is not None:
            self._event_bus.publish("ops.lifecycle.changed", {"service_name": service_name, "state": state.value, "error": error})
