from __future__ import annotations

from datetime import datetime, timezone

from .models import DependencyHealthProbe, HealthStatus, OperationalHealthStatus, ServiceHealthProbe, ServiceLifecycleRecord, ServiceLifecycleState
from .telemetry import TelemetryRecorder


def map_health_status(status: HealthStatus | str) -> OperationalHealthStatus:
    raw = status.value if isinstance(status, HealthStatus) else str(status)
    return {
        "ready": OperationalHealthStatus.READY,
        "degraded": OperationalHealthStatus.DEGRADED,
        "stopped": OperationalHealthStatus.STOPPED,
        "failed": OperationalHealthStatus.FAILED,
        "recovering": OperationalHealthStatus.RECOVERING,
        "starting": OperationalHealthStatus.RECOVERING,
    }.get(raw, OperationalHealthStatus.DEGRADED)


class HealthcheckAggregator:
    def __init__(self, telemetry: TelemetryRecorder, resilience_controller=None) -> None:
        self._telemetry = telemetry
        self._resilience = resilience_controller

    def build_probe(self, *, service_status, lifecycle: ServiceLifecycleRecord | None = None) -> ServiceHealthProbe:
        dependencies: list[DependencyHealthProbe] = []
        breakers = self._resilience.breaker_snapshot() if self._resilience is not None else []
        for item in breakers:
            if item["service_name"] != service_status.name:
                continue
            dependencies.append(
                DependencyHealthProbe(
                    service_name=service_status.name,
                    dependency_name=str(item["dependency_name"]),
                    status=OperationalHealthStatus.DEGRADED if item["state"] != "closed" else OperationalHealthStatus.READY,
                    failures_recent=int(item["failure_count"]),
                    metadata=item,
                )
            )
        status = map_health_status(service_status.status)
        if lifecycle is not None and lifecycle.state == ServiceLifecycleState.FAILED:
            status = OperationalHealthStatus.FAILED
        elif lifecycle is not None and lifecycle.state == ServiceLifecycleState.RECOVERING:
            status = OperationalHealthStatus.RECOVERING
        return ServiceHealthProbe(
            service_name=service_status.name,
            liveness=service_status.status not in {HealthStatus.STOPPED, HealthStatus.FAILED},
            readiness=service_status.status in {HealthStatus.READY, HealthStatus.DEGRADED},
            status=status,
            checked_at=datetime.now(timezone.utc),
            failures_recent=self._telemetry.failures_recent(service_status.name),
            warnings=list(service_status.details.get("warnings", [])) if isinstance(service_status.details, dict) else [],
            dependencies=dependencies,
            metadata=service_status.details if isinstance(service_status.details, dict) else {},
        )

    def aggregate_status(self, probes: list[ServiceHealthProbe]) -> OperationalHealthStatus:
        if any(item.status == OperationalHealthStatus.FAILED for item in probes):
            return OperationalHealthStatus.FAILED
        if any(item.status in {OperationalHealthStatus.DEGRADED, OperationalHealthStatus.RECOVERING} for item in probes):
            return OperationalHealthStatus.DEGRADED
        if probes and all(item.status == OperationalHealthStatus.STOPPED for item in probes):
            return OperationalHealthStatus.STOPPED
        return OperationalHealthStatus.READY
