from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class HealthStatus(StrEnum):
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class TaskLifecycleStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationStatus(StrEnum):
    RUNNING = "running"
    DEFERRED = "deferred"
    REJECTED = "rejected"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ServiceStatus(JarvisBaseModel):
    name: str
    status: HealthStatus
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class RuntimeTaskRecord(JarvisBaseModel):
    task_id: str
    route_type: str
    target: str
    source: str
    status: TaskLifecycleStatus
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_summary: str | None = None
    error: dict[str, Any] | None = None


class ToolInvocationRecord(JarvisBaseModel):
    correlation_id: str
    tool_name: str
    status: str
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class ModelInvocationRecord(JarvisBaseModel):
    correlation_id: str
    provider: str
    provider_kind: str | None = None
    logical_model: str
    model_name: str
    task_type: str
    status: str
    latency_ms: float | None = None
    fallback_used: bool = False
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class EmbeddingInvocationRecord(JarvisBaseModel):
    correlation_id: str
    provider: str
    provider_kind: str | None = None
    logical_model: str
    model_name: str
    task_type: str
    status: str
    latency_ms: float | None = None
    fallback_used: bool = False
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class UIAutomationRecord(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    risk_level: str
    status: str
    window_title: str | None = None
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class VoiceInvocationRecord(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    provider: str | None = None
    backend: str | None = None
    status: str
    latency_ms: float | None = None
    session_id: str | None = None
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class VisionInvocationRecord(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    backend: str | None = None
    provider: str | None = None
    analyzer: str | None = None
    status: str
    latency_ms: float | None = None
    capture_target: str | None = None
    fallback_used: bool = False
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class AutonomyInvocationRecord(JarvisBaseModel):
    mission_id: str
    operation_name: str
    status: str
    autonomy_level: str | None = None
    step_id: str | None = None
    goal: str | None = None
    stop_reason: str | None = None
    invoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


class ModeSnapshot(JarvisBaseModel):
    active_mode: str
    previous_mode: str | None = None
    sticky: bool = True
    changed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str | None = None


class RuntimeSnapshot(JarvisBaseModel):
    app_name: str
    environment: str
    mode: ModeSnapshot
    services: list[ServiceStatus] = Field(default_factory=list)
    active_tasks: list[RuntimeTaskRecord] = Field(default_factory=list)
    recent_tasks: list[RuntimeTaskRecord] = Field(default_factory=list)
    recent_tool_invocations: list[ToolInvocationRecord] = Field(default_factory=list)
    recent_model_invocations: list[ModelInvocationRecord] = Field(default_factory=list)
    recent_embedding_invocations: list[EmbeddingInvocationRecord] = Field(default_factory=list)
    recent_ui_operations: list[UIAutomationRecord] = Field(default_factory=list)
    recent_voice_invocations: list[VoiceInvocationRecord] = Field(default_factory=list)
    recent_vision_invocations: list[VisionInvocationRecord] = Field(default_factory=list)
    recent_autonomy_receipts: list[AutonomyInvocationRecord] = Field(default_factory=list)
    action_names: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)


class OperationalHealthStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class CircuitBreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ServiceLifecycleState(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    READY = "ready"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RECOVERING = "recovering"


class DependencyHealthProbe(JarvisBaseModel):
    service_name: str
    dependency_name: str
    status: OperationalHealthStatus
    latency_ms: float | None = None
    failures_recent: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ServiceHealthProbe(JarvisBaseModel):
    service_name: str
    liveness: bool
    readiness: bool
    status: OperationalHealthStatus
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None
    failures_recent: int = 0
    warnings: list[str] = Field(default_factory=list)
    dependencies: list[DependencyHealthProbe] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationalIssue(JarvisBaseModel):
    issue_id: str
    service_name: str
    severity: str
    summary: str
    symptom: str | None = None
    probable_cause: str | None = None
    dependency_name: str | None = None
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticReport(JarvisBaseModel):
    service_name: str
    status: OperationalHealthStatus
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    issues: list[OperationalIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dependencies: list[DependencyHealthProbe] = Field(default_factory=list)
    recent_errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureRecord(JarvisBaseModel):
    record_id: str
    service_name: str
    operation_name: str
    dependency_name: str | None = None
    error: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlowOperationRecord(JarvisBaseModel):
    record_id: str
    service_name: str
    operation_name: str
    dependency_name: str | None = None
    latency_ms: float
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryRecord(JarvisBaseModel):
    record_id: str
    service_name: str
    success: bool
    message: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdmissionDecision(JarvisBaseModel):
    service_name: str
    granted: bool
    deferred: bool = False
    reason: str | None = None
    active_count: int = 0
    queue_depth: int = 0
    limit: int = 0


class ActiveOperationRecord(JarvisBaseModel):
    operation_id: str
    correlation_id: str
    service_name: str
    operation_name: str
    status: OperationStatus
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    deadline_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    timeout_ms: int | None = None
    watchdog_timeout_ms: int | None = None
    timeout_hard: bool = False
    progress_message: str | None = None
    last_error: str | None = None
    admission: AdmissionDecision | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceSample(JarvisBaseModel):
    sampled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float | None = None
    process_cpu_percent: float | None = None
    ram_percent: float | None = None
    ram_used_bytes: int | None = None
    ram_available_bytes: int | None = None
    process_rss_bytes: int | None = None
    disk_total_bytes: int | None = None
    disk_used_bytes: int | None = None
    disk_free_bytes: int | None = None
    disk_percent: float | None = None
    temperature_celsius: float | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AutoRecoveryPolicy(JarvisBaseModel):
    enabled: bool = False
    cooldown_seconds: float = 60.0
    max_attempts_per_window: int = 2
    window_seconds: float = 300.0


class ServiceLifecycleRecord(JarvisBaseModel):
    service_name: str
    state: ServiceLifecycleState
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    start_count: int = 0
    stop_count: int = 0
    recover_count: int = 0
    last_error: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResilienceExecutionReceipt(JarvisBaseModel):
    service_name: str
    operation_name: str
    dependency_name: str | None = None
    success: bool
    breaker_state: CircuitBreakerState
    attempt_count: int = 1
    timeout_ms: int | None = None
    latency_ms: float | None = None
    timed_out: bool = False
    retry_budget_consumed: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ServiceOperationalSnapshot(JarvisBaseModel):
    service_name: str
    lifecycle: ServiceLifecycleState
    health: ServiceHealthProbe
    diagnostics: DiagnosticReport | None = None
    breaker_states: list[dict[str, Any]] = Field(default_factory=list)
    retry_budgets: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationalSnapshot(JarvisBaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    app_name: str
    environment: str
    mode: ModeSnapshot
    services: list[ServiceOperationalSnapshot] = Field(default_factory=list)
    aggregate_status: OperationalHealthStatus = OperationalHealthStatus.READY
    degraded_dependencies: list[str] = Field(default_factory=list)
    recent_failures: list[FailureRecord] = Field(default_factory=list)
    recent_slow_operations: list[SlowOperationRecord] = Field(default_factory=list)
    recent_recoveries: list[RecoveryRecord] = Field(default_factory=list)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeDiagnosticsSnapshot(JarvisBaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    app_name: str
    environment: str
    reports: list[DiagnosticReport] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
