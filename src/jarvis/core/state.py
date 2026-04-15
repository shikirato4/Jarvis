from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from .models import (
    FailureRecord,
    AutonomyInvocationRecord,
    EmbeddingInvocationRecord,
    HealthStatus,
    ModelInvocationRecord,
    OperationalSnapshot,
    RecoveryRecord,
    RuntimeSnapshot,
    RuntimeTaskRecord,
    ServiceStatus,
    SlowOperationRecord,
    TaskLifecycleStatus,
    ToolInvocationRecord,
    UIAutomationRecord,
    VisionInvocationRecord,
    VoiceInvocationRecord,
)
from .modes import ModeManager


class RuntimeStateManager:
    def __init__(self, app_name: str, environment: str, mode_manager: ModeManager, history_limit: int = 50) -> None:
        self._app_name = app_name
        self._environment = environment
        self._mode_manager = mode_manager
        self._history_limit = history_limit
        self._lock = RLock()
        self._services: dict[str, ServiceStatus] = {}
        self._active_tasks: dict[str, RuntimeTaskRecord] = {}
        self._recent_tasks: deque[RuntimeTaskRecord] = deque(maxlen=history_limit)
        self._recent_tools: deque[ToolInvocationRecord] = deque(maxlen=history_limit)
        self._recent_models: deque[ModelInvocationRecord] = deque(maxlen=history_limit)
        self._recent_embeddings: deque[EmbeddingInvocationRecord] = deque(maxlen=history_limit)
        self._recent_ui: deque[UIAutomationRecord] = deque(maxlen=history_limit)
        self._recent_voice: deque[VoiceInvocationRecord] = deque(maxlen=history_limit)
        self._recent_vision: deque[VisionInvocationRecord] = deque(maxlen=history_limit)
        self._recent_autonomy: deque[AutonomyInvocationRecord] = deque(maxlen=history_limit)
        self._recent_failures: deque[FailureRecord] = deque(maxlen=history_limit)
        self._recent_slow_operations: deque[SlowOperationRecord] = deque(maxlen=history_limit)
        self._recent_recoveries: deque[RecoveryRecord] = deque(maxlen=history_limit)
        self._ops_snapshots: deque[OperationalSnapshot] = deque(maxlen=history_limit)

    def bind(self, event_bus: "EventBus") -> None:
        event_bus.subscribe("tool.executed", self._on_tool_executed)
        event_bus.subscribe("tool.failed", self._on_tool_failed)
        event_bus.subscribe("model.executed", self._on_model_executed)
        event_bus.subscribe("model.failed", self._on_model_failed)
        event_bus.subscribe("embedding.executed", self._on_embedding_executed)
        event_bus.subscribe("embedding.failed", self._on_embedding_failed)
        event_bus.subscribe("ui.executed", self._on_ui_executed)
        event_bus.subscribe("ui.failed", self._on_ui_failed)
        event_bus.subscribe("voice.executed", self._on_voice_executed)
        event_bus.subscribe("voice.failed", self._on_voice_failed)
        event_bus.subscribe("vision.executed", self._on_vision_executed)
        event_bus.subscribe("vision.failed", self._on_vision_failed)
        event_bus.subscribe("autonomy.started", self._on_autonomy_started)
        event_bus.subscribe("autonomy.step_planned", self._on_autonomy_updated)
        event_bus.subscribe("autonomy.step_executed", self._on_autonomy_updated)
        event_bus.subscribe("autonomy.verified", self._on_autonomy_updated)
        event_bus.subscribe("autonomy.replanned", self._on_autonomy_updated)
        event_bus.subscribe("autonomy.stopped", self._on_autonomy_updated)
        event_bus.subscribe("autonomy.failed", self._on_autonomy_updated)
        event_bus.subscribe("ops.failure_recorded", self._on_ops_failure)
        event_bus.subscribe("ops.slow_operation_recorded", self._on_ops_slow)
        event_bus.subscribe("ops.recovery_recorded", self._on_ops_recovery)

    def register_service(self, name: str, status: HealthStatus = HealthStatus.STOPPED, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._services[name] = ServiceStatus(name=name, status=status, details=details or {})

    def update_service(self, name: str, status: HealthStatus, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._services[name] = ServiceStatus(name=name, status=status, details=details or {})

    def begin_task(
        self,
        *,
        task_id: str,
        route_type: str,
        target: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeTaskRecord:
        record = RuntimeTaskRecord(
            task_id=task_id,
            route_type=route_type,
            target=target,
            source=source,
            status=TaskLifecycleStatus.RUNNING,
            metadata=metadata or {},
        )
        with self._lock:
            self._active_tasks[task_id] = record
        return record

    def complete_task(self, task_id: str, *, output_summary: str | None = None) -> None:
        with self._lock:
            record = self._active_tasks.pop(task_id, None)
            if record is None:
                return
            record.status = TaskLifecycleStatus.COMPLETED
            record.finished_at = datetime.now(timezone.utc)
            record.output_summary = output_summary
            self._recent_tasks.appendleft(record)

    def fail_task(self, task_id: str, *, error: dict[str, Any]) -> None:
        with self._lock:
            record = self._active_tasks.pop(task_id, None)
            if record is None:
                return
            record.status = TaskLifecycleStatus.FAILED
            record.finished_at = datetime.now(timezone.utc)
            record.error = error
            self._recent_tasks.appendleft(record)

    def snapshot(self, *, action_names: list[str], tool_names: list[str], include_history: bool = True) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                app_name=self._app_name,
                environment=self._environment,
                mode=self._mode_manager.snapshot(),
                services=list(self._services.values()),
                active_tasks=list(self._active_tasks.values()),
                recent_tasks=list(self._recent_tasks) if include_history else [],
                recent_tool_invocations=list(self._recent_tools) if include_history else [],
                recent_model_invocations=list(self._recent_models) if include_history else [],
                recent_embedding_invocations=list(self._recent_embeddings) if include_history else [],
                recent_ui_operations=list(self._recent_ui) if include_history else [],
                recent_voice_invocations=list(self._recent_voice) if include_history else [],
                recent_vision_invocations=list(self._recent_vision) if include_history else [],
                recent_autonomy_receipts=list(self._recent_autonomy) if include_history else [],
                action_names=sorted(action_names),
                tool_names=sorted(tool_names),
            )

    def record_operational_snapshot(self, snapshot: OperationalSnapshot) -> None:
        with self._lock:
            self._ops_snapshots.appendleft(snapshot)

    def operational_snapshots(self) -> list[OperationalSnapshot]:
        with self._lock:
            return list(self._ops_snapshots)

    def recent_failures(self) -> list[FailureRecord]:
        with self._lock:
            return list(self._recent_failures)

    def recent_slow_operations(self) -> list[SlowOperationRecord]:
        with self._lock:
            return list(self._recent_slow_operations)

    def recent_recoveries(self) -> list[RecoveryRecord]:
        with self._lock:
            return list(self._recent_recoveries)

    def trim_history(self, *, receipt_limit: int, snapshot_limit: int, max_age_seconds: float | None = None) -> dict[str, int]:
        with self._lock:
            if max_age_seconds is not None:
                cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
                for queue in (
                    self._recent_tasks,
                    self._recent_tools,
                    self._recent_models,
                    self._recent_embeddings,
                    self._recent_ui,
                    self._recent_voice,
                    self._recent_vision,
                    self._recent_autonomy,
                    self._recent_failures,
                    self._recent_slow_operations,
                    self._recent_recoveries,
                ):
                    while queue and getattr(queue[-1], "finished_at", None) is not None and queue[-1].finished_at is not None and queue[-1].finished_at.timestamp() < cutoff:
                        queue.pop()
                    while queue and getattr(queue[-1], "invoked_at", None) is not None and queue[-1].invoked_at.timestamp() < cutoff:
                        queue.pop()
                    while queue and getattr(queue[-1], "recorded_at", None) is not None and queue[-1].recorded_at.timestamp() < cutoff:
                        queue.pop()
                while self._ops_snapshots and self._ops_snapshots[-1].timestamp.timestamp() < cutoff:
                    self._ops_snapshots.pop()
            before_receipts = sum(
                len(queue)
                for queue in (
                    self._recent_tasks,
                    self._recent_tools,
                    self._recent_models,
                    self._recent_embeddings,
                    self._recent_ui,
                    self._recent_voice,
                    self._recent_vision,
                    self._recent_autonomy,
                    self._recent_failures,
                    self._recent_slow_operations,
                    self._recent_recoveries,
                )
            )
            before_snapshots = len(self._ops_snapshots)
            for queue in (
                self._recent_tasks,
                self._recent_tools,
                self._recent_models,
                self._recent_embeddings,
                self._recent_ui,
                self._recent_voice,
                self._recent_vision,
                self._recent_autonomy,
                self._recent_failures,
                self._recent_slow_operations,
                self._recent_recoveries,
            ):
                while len(queue) > receipt_limit:
                    queue.pop()
            while len(self._ops_snapshots) > snapshot_limit:
                self._ops_snapshots.pop()
            after_receipts = sum(
                len(queue)
                for queue in (
                    self._recent_tasks,
                    self._recent_tools,
                    self._recent_models,
                    self._recent_embeddings,
                    self._recent_ui,
                    self._recent_voice,
                    self._recent_vision,
                    self._recent_autonomy,
                    self._recent_failures,
                    self._recent_slow_operations,
                    self._recent_recoveries,
                )
            )
            return {
                "receipts_trimmed": max(before_receipts - after_receipts, 0),
                "snapshots_trimmed": max(before_snapshots - len(self._ops_snapshots), 0),
            }

    def _on_tool_executed(self, payload: dict[str, Any]) -> None:
        self._record_tool(payload)

    def _on_tool_failed(self, payload: dict[str, Any]) -> None:
        self._record_tool(payload)

    def _on_model_executed(self, payload: dict[str, Any]) -> None:
        self._record_model(payload, status="success")

    def _on_model_failed(self, payload: dict[str, Any]) -> None:
        self._record_model(payload, status="failed")

    def _on_embedding_executed(self, payload: dict[str, Any]) -> None:
        self._record_embedding(payload, status="success")

    def _on_embedding_failed(self, payload: dict[str, Any]) -> None:
        self._record_embedding(payload, status="failed")

    def _on_ui_executed(self, payload: dict[str, Any]) -> None:
        self._record_ui(payload, status="success")

    def _on_ui_failed(self, payload: dict[str, Any]) -> None:
        self._record_ui(payload, status="failed")

    def _on_voice_executed(self, payload: dict[str, Any]) -> None:
        self._record_voice(payload, status="success")

    def _on_voice_failed(self, payload: dict[str, Any]) -> None:
        self._record_voice(payload, status="failed")

    def _on_vision_executed(self, payload: dict[str, Any]) -> None:
        self._record_vision(payload, status="success")

    def _on_vision_failed(self, payload: dict[str, Any]) -> None:
        self._record_vision(payload, status="failed")

    def _on_autonomy_started(self, payload: dict[str, Any]) -> None:
        self._record_autonomy(payload, status="started")

    def _on_autonomy_updated(self, payload: dict[str, Any]) -> None:
        self._record_autonomy(payload, status=str(payload.get("status", "updated")))

    def _record_tool(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_tools.appendleft(
                ToolInvocationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    tool_name=str(payload.get("tool", "")),
                    status=str(payload.get("status", "")),
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_model(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_models.appendleft(
                ModelInvocationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    provider=str(payload.get("provider", "")),
                    provider_kind=str(payload.get("provider_kind")) if payload.get("provider_kind") is not None else None,
                    logical_model=str(payload.get("logical_model", "")),
                    model_name=str(payload.get("model_name", "")),
                    task_type=str(payload.get("task_type", "")),
                    status=status,
                    latency_ms=float(payload.get("latency_ms")) if payload.get("latency_ms") is not None else None,
                    fallback_used=bool(payload.get("fallback_used", False)),
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_embedding(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_embeddings.appendleft(
                EmbeddingInvocationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    provider=str(payload.get("provider", "")),
                    provider_kind=str(payload.get("provider_kind")) if payload.get("provider_kind") is not None else None,
                    logical_model=str(payload.get("logical_model", "")),
                    model_name=str(payload.get("model_name", "")),
                    task_type=str(payload.get("task_type", "")),
                    status=status,
                    latency_ms=float(payload.get("latency_ms")) if payload.get("latency_ms") is not None else None,
                    fallback_used=bool(payload.get("fallback_used", False)),
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_ui(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_ui.appendleft(
                UIAutomationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    operation_name=str(payload.get("operation_name", "")),
                    risk_level=str(payload.get("risk_level", "")),
                    status=status,
                    window_title=str(payload.get("window_title")) if payload.get("window_title") is not None else None,
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_voice(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_voice.appendleft(
                VoiceInvocationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    operation_name=str(payload.get("operation_name", "")),
                    provider=str(payload.get("provider")) if payload.get("provider") is not None else None,
                    backend=str(payload.get("backend")) if payload.get("backend") is not None else None,
                    status=status,
                    latency_ms=float(payload.get("latency_ms")) if payload.get("latency_ms") is not None else None,
                    session_id=str(payload.get("session_id")) if payload.get("session_id") is not None else None,
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_vision(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_vision.appendleft(
                VisionInvocationRecord(
                    correlation_id=str(payload.get("correlation_id", "")),
                    operation_name=str(payload.get("operation_name", "")),
                    backend=str(payload.get("backend")) if payload.get("backend") is not None else None,
                    provider=str(payload.get("provider")) if payload.get("provider") is not None else None,
                    analyzer=str(payload.get("analyzer")) if payload.get("analyzer") is not None else None,
                    status=status,
                    latency_ms=float(payload.get("latency_ms")) if payload.get("latency_ms") is not None else None,
                    capture_target=str(payload.get("capture_target")) if payload.get("capture_target") is not None else None,
                    fallback_used=bool(payload.get("fallback_used", False)),
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _record_autonomy(self, payload: dict[str, Any], *, status: str) -> None:
        with self._lock:
            self._recent_autonomy.appendleft(
                AutonomyInvocationRecord(
                    mission_id=str(payload.get("mission_id", "")),
                    operation_name=str(payload.get("operation_name", "")),
                    status=status,
                    autonomy_level=str(payload.get("autonomy_level")) if payload.get("autonomy_level") is not None else None,
                    step_id=str(payload.get("step_id")) if payload.get("step_id") is not None else None,
                    goal=str(payload.get("goal")) if payload.get("goal") is not None else None,
                    stop_reason=str(payload.get("stop_reason")) if payload.get("stop_reason") is not None else None,
                    data=payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {},
                )
            )

    def _on_ops_failure(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_failures.appendleft(FailureRecord.model_validate(payload))

    def _on_ops_slow(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_slow_operations.appendleft(SlowOperationRecord.model_validate(payload))

    def _on_ops_recovery(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_recoveries.appendleft(RecoveryRecord.model_validate(payload))


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.core.events import EventBus
