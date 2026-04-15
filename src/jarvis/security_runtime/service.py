from __future__ import annotations

from jarvis.core.models import HealthStatus, ServiceStatus

from .analyzer import analyze_security, check_password
from .base import SecurityAnalyzeRequest, SecurityPasswordCheckRequest, SecurityResult


class SecurityRuntimeService:
    service_name = "security_runtime"

    def __init__(self, settings, *, logger=None) -> None:
        self._settings = settings
        self._logger = logger
        self._started = False

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.service_name,
            status=HealthStatus.READY if self._started else HealthStatus.STOPPED,
            details=self.status(),
        )

    def status(self) -> dict[str, object]:
        return {
            "started": self._started,
            "capabilities": [
                "python_static_analysis",
                "workspace_self_audit",
                "password_strength_review",
                "ethical_security_education",
            ],
        }

    def analyze(self, request: SecurityAnalyzeRequest) -> SecurityResult:
        return analyze_security(request, workspace_root=self._settings.resolved_workspace_root)

    def check_password(self, request: SecurityPasswordCheckRequest) -> SecurityResult:
        return check_password(request)
