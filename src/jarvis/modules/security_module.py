from __future__ import annotations

from pydantic import BaseModel

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.security_runtime import SecurityAnalyzeRequest, SecurityPasswordCheckRequest


class SecurityAnalyzePayload(BaseModel):
    query: str | None = None
    code: str | None = None
    path: str | None = None
    include_workspace: bool = False
    max_findings: int = 50
    audit_kind: str | None = None
    host: str = "127.0.0.1"
    ports: list[int] = []
    timeout_ms: int = 250


class SecurityPasswordPayload(BaseModel):
    password: str


class SecurityModule:
    name = "security"
    description = "Ethical cybersecurity runtime for defensive analysis and education."

    def __init__(self, security_runtime) -> None:
        self._security = security_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="security.analyze",
                description="Analyze code or local workspace security posture.",
                payload_model=SecurityAnalyzePayload,
                handler=self._analyze,
                tags=("security", "analysis", "ethical"),
            )
        )
        registry.register(
            ActionDefinition(
                name="security.check_password",
                description="Evaluate password strength defensively.",
                payload_model=SecurityPasswordPayload,
                handler=self._check_password,
                tags=("security", "password", "ethical"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="security.runtime",
                module_name=self.name,
                intent="security",
                description="Ethical cybersecurity analysis, code review and defensive education.",
                action_names=("security.analyze", "security.check_password"),
                keywords=(
                    "seguridad",
                    "hack",
                    "contraseña",
                    "contrasena",
                    "vulnerabilidad",
                    "owasp",
                    "criptografia",
                    "criptografía",
                    "xss",
                    "sqli",
                    "ciberseguridad",
                ),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="analysis",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _analyze(self, context: ActionContext, payload: SecurityAnalyzePayload) -> ActionResult:
        result = self._security.analyze(
            SecurityAnalyzeRequest(
                query=payload.query,
                code=payload.code,
                path=payload.path,
                include_workspace=payload.include_workspace,
                max_findings=payload.max_findings,
                audit_kind=payload.audit_kind,
                host=payload.host,
                ports=payload.ports,
                timeout_ms=payload.timeout_ms,
            )
        )
        return ActionResult(message="security analysis completed", data=result.model_dump(mode="json"))

    def _check_password(self, context: ActionContext, payload: SecurityPasswordPayload) -> ActionResult:
        result = self._security.check_password(SecurityPasswordCheckRequest(password=payload.password))
        return ActionResult(message="password review completed", data=result.model_dump(mode="json"))

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        lowered = (request.query or "").casefold()
        payload = dict(request.payload)
        if "password" in payload or "contrase" in lowered:
            if "password" not in payload and request.query:
                payload["password"] = request.query.split()[-1]
            return [ActionStep(action="security.check_password", payload=payload)]
        payload.setdefault("query", request.query)
        return [ActionStep(action="security.analyze", payload=payload)]
