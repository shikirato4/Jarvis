from __future__ import annotations

from uuid import uuid4

from .models import DiagnosticReport, OperationalHealthStatus, OperationalIssue, RuntimeDiagnosticsSnapshot, ServiceHealthProbe


class DiagnosticsAggregator:
    def __init__(self, telemetry, event_bus=None, resilience_controller=None) -> None:
        self._telemetry = telemetry
        self._event_bus = event_bus
        self._resilience = resilience_controller

    def build_report(self, *, service_name: str, probe: ServiceHealthProbe) -> DiagnosticReport:
        issues: list[OperationalIssue] = []
        if probe.status in {OperationalHealthStatus.DEGRADED, OperationalHealthStatus.FAILED, OperationalHealthStatus.RECOVERING}:
            issues.append(
                OperationalIssue(
                    issue_id=str(uuid4()),
                    service_name=service_name,
                    severity="high" if probe.status == OperationalHealthStatus.FAILED else "medium",
                    summary=f"{service_name} is {probe.status.value}",
                    probable_cause="recent failures or degraded dependencies",
                    metadata={"failures_recent": probe.failures_recent},
                )
            )
        for dependency in probe.dependencies:
            if dependency.status != OperationalHealthStatus.READY:
                issues.append(
                    OperationalIssue(
                        issue_id=str(uuid4()),
                        service_name=service_name,
                        severity="medium",
                        summary=f"dependency '{dependency.dependency_name}' is {dependency.status.value}",
                        probable_cause="breaker opened or repeated failures",
                        dependency_name=dependency.dependency_name,
                        metadata=dependency.metadata,
                    )
                )
        recent_errors = [item.error for item in self._telemetry.recent_failures() if item.service_name == service_name][:10]
        warnings = list(probe.warnings)
        if recent_errors and "recent failures detected" not in warnings:
            warnings.append("recent failures detected")
        return DiagnosticReport(
            service_name=service_name,
            status=probe.status,
            issues=issues,
            warnings=warnings,
            dependencies=probe.dependencies,
            recent_errors=recent_errors,
            metadata=probe.metadata,
        )

    def snapshot(self, *, app_name: str, environment: str, reports: list[DiagnosticReport]) -> RuntimeDiagnosticsSnapshot:
        return RuntimeDiagnosticsSnapshot(app_name=app_name, environment=environment, reports=reports)
