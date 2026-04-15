from __future__ import annotations

from jarvis.core.diagnostics import DiagnosticsAggregator
from jarvis.core.healthchecks import HealthcheckAggregator
from jarvis.core.lifecycle import RecoveryPlan, RuntimeLifecycleSupervisor
from jarvis.core.snapshots import SnapshotBuilder
from jarvis.core.models import DiagnosticReport, ServiceOperationalSnapshot


class OpsRuntimeService:
    service_name = "ops_runtime"

    def __init__(
        self,
        *,
        settings,
        mode_manager,
        state_manager,
        event_bus,
        telemetry,
        resilience_controller,
        lifecycle_supervisor: RuntimeLifecycleSupervisor,
        retention_manager,
        operation_registry,
        operation_watchdog,
        resource_monitor,
        admission_controller,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._state_manager = state_manager
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._resilience = resilience_controller
        self._lifecycle = lifecycle_supervisor
        self._retention = retention_manager
        self._operations = operation_registry
        self._watchdog = operation_watchdog
        self._resources = resource_monitor
        self._admission = admission_controller
        self._health = HealthcheckAggregator(telemetry, resilience_controller=resilience_controller)
        self._diagnostics = DiagnosticsAggregator(telemetry, event_bus=event_bus, resilience_controller=resilience_controller)
        self._snapshots = SnapshotBuilder()
        self._started = False

    def start(self) -> None:
        self._started = True
        self._resources.start()
        self._watchdog.start()

    def stop(self) -> None:
        self._watchdog.stop()
        self._resources.stop()
        self._started = False

    def status(self) -> dict[str, object]:
        snapshot = self.snapshot()
        return {
            "started": self._started,
            "aggregate_status": snapshot.aggregate_status.value,
            "service_count": len(snapshot.services),
            "degraded_dependencies": snapshot.degraded_dependencies,
            "telemetry": snapshot.telemetry,
            "resources": snapshot.metadata.get("resources", {}),
            "operations": snapshot.metadata.get("operations", {}),
            "queues": snapshot.metadata.get("queues", []),
        }

    def health(self) -> list:
        runtime = self._state_manager.snapshot(action_names=[], tool_names=[], include_history=False)
        lifecycle_map = {item.service_name: item for item in self._lifecycle.records()}
        return [self._health.build_probe(service_status=service, lifecycle=lifecycle_map.get(service.name)) for service in runtime.services]

    def diagnostics(self, service_name: str | None = None):
        probes = self.health()
        reports = [self._diagnostics.build_report(service_name=probe.service_name, probe=probe) for probe in probes]
        if service_name is not None:
            reports = [report for report in reports if report.service_name == service_name]
        return reports

    def snapshot(self):
        runtime = self._state_manager.snapshot(action_names=[], tool_names=[], include_history=False)
        lifecycle_map = {item.service_name: item for item in self._lifecycle.records()}
        probes = self.health()
        reports_map: dict[str, DiagnosticReport] = {report.service_name: report for report in self.diagnostics()}
        service_snapshots = [
            ServiceOperationalSnapshot(
                service_name=probe.service_name,
                lifecycle=(lifecycle_map.get(probe.service_name).state if lifecycle_map.get(probe.service_name) is not None else "created"),
                health=probe,
                diagnostics=reports_map.get(probe.service_name),
                breaker_states=[item for item in self._resilience.breaker_snapshot() if item["service_name"] == probe.service_name],
                retry_budgets=[item for item in self._resilience.retry_budget_snapshot() if item["service_name"] == probe.service_name],
            )
            for probe in probes
        ]
        snapshot = self._snapshots.build_operational_snapshot(
            app_name=runtime.app_name,
            environment=runtime.environment,
            mode=runtime.mode,
            service_snapshots=service_snapshots,
            aggregate_status=self._health.aggregate_status(probes),
            recent_failures=self._telemetry.recent_failures(),
            recent_slow_operations=self._telemetry.recent_slow_operations(),
            recent_recoveries=self._telemetry.recent_recoveries(),
            recent_events=self._event_bus.recent_events(),
            telemetry=self._telemetry.snapshot(),
            metadata={
                "resources": self._resources.snapshot(),
                "operations": self._operations.snapshot(),
                "queues": self._admission.snapshot(),
            },
        )
        self._state_manager.record_operational_snapshot(snapshot)
        return snapshot

    def recover_service(self, service_name: str, *, dry_run: bool = False):
        return self._lifecycle.recover_service(RecoveryPlan(service_name=service_name, dry_run=dry_run))

    def reset_breaker(self, service_name: str, dependency_name: str | None = None) -> dict[str, object]:
        return {"reset": self._resilience.reset_breaker(service_name, dependency_name), "service_name": service_name, "dependency_name": dependency_name}

    def retention_sweep(self):
        return self._retention.sweep()

    def operations(self) -> dict[str, object]:
        return self._operations.snapshot()

    def resources(self) -> dict[str, object]:
        return self._resources.snapshot()

    def cancel_operation(self, operation_id: str, *, reason: str = "cancel requested from ops") -> dict[str, object]:
        return {"cancelled": self._operations.cancel(operation_id, reason=reason), "operation_id": operation_id, "reason": reason}
