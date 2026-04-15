from __future__ import annotations

from .models import OperationalSnapshot, ServiceOperationalSnapshot


class SnapshotBuilder:
    def build_operational_snapshot(
        self,
        *,
        app_name: str,
        environment: str,
        mode,
        service_snapshots: list[ServiceOperationalSnapshot],
        aggregate_status,
        recent_failures,
        recent_slow_operations,
        recent_recoveries,
        recent_events,
        telemetry,
        metadata: dict[str, object] | None = None,
    ) -> OperationalSnapshot:
        degraded_dependencies: list[str] = []
        for service in service_snapshots:
            for dependency in service.health.dependencies:
                if dependency.status.value != "ready":
                    degraded_dependencies.append(f"{service.service_name}:{dependency.dependency_name}")
        return OperationalSnapshot(
            app_name=app_name,
            environment=environment,
            mode=mode,
            services=service_snapshots,
            aggregate_status=aggregate_status,
            degraded_dependencies=degraded_dependencies,
            recent_failures=recent_failures,
            recent_slow_operations=recent_slow_operations,
            recent_recoveries=recent_recoveries,
            recent_events=recent_events,
            telemetry=telemetry,
            metadata=metadata or {},
        )
