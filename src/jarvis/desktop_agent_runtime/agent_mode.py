from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentMode(StrEnum):
    OBSERVE_ONLY = "observe_only"
    ASSIST = "assist"
    GUIDED_CONTROL = "guided_control"
    LIMITED_AUTOPILOT = "limited_autopilot"


class AgentStatus(StrEnum):
    DISABLED = "disabled"
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    WAITING_CONFIRMATION = "waiting_confirmation"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


class AgentSafetyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_STRONG_CONFIRMATION = "require_strong_confirmation"
    BLOCK = "block"


@dataclass
class AgentObservation:
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class AgentAction:
    action_type: str
    title: str = ""
    description: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    risk: AgentRisk = AgentRisk.LOW

    def searchable_text(self) -> str:
        pieces = [self.action_type, self.title, self.description, str(self.payload)]
        return " ".join(item for item in pieces if item).casefold()


@dataclass
class AgentStep:
    step_id: str
    title: str
    action: AgentAction
    status: AgentStatus = AgentStatus.IDLE


@dataclass
class AgentVerification:
    status: str
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAuditLog:
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, event_type: str, *, status: str, detail: str = "", metadata: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "status": status,
                "detail": detail,
                "metadata": metadata or {},
                "created_at": _utcnow().isoformat(),
            }
        )


@dataclass
class AgentSession:
    user_goal: str
    mode: AgentMode = AgentMode.GUIDED_CONTROL
    session_id: str = field(default_factory=lambda: str(uuid4()))
    status: AgentStatus = AgentStatus.IDLE
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    observations: list[AgentObservation] = field(default_factory=list)
    plan: list[AgentStep] = field(default_factory=list)
    current_step: AgentStep | None = None
    confirmations: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    audit: AgentAuditLog = field(default_factory=AgentAuditLog)


@dataclass(frozen=True)
class AgentSafetyResult:
    decision: AgentSafetyDecision
    risk: AgentRisk
    reason: str
    requires_pin: bool = False


class AgentSafetyGate:
    _BLOCKED_TERMS = (
        "password",
        "contraseña",
        "token",
        ".env",
        "cookie",
        "credential",
        "credencial",
        "private key",
        "api key",
        "banco",
        "bank",
        "comprar",
        "compra",
        "pagar",
        "pago",
        "malware",
        "keylogger",
        "stealer",
        "rat",
        "evadir",
        "bypass",
        "robar",
        "extraer claves",
    )
    _HIGH_TERMS = (
        "borrar",
        "delete",
        "remove",
        "instalar",
        "install",
        "ejecutar script",
        "run script",
        "registro",
        "registry",
        "system32",
        "firewall",
        "antivirus",
        "driver",
        "formulario",
        "publicar",
        "correo",
    )
    _MEDIUM_ACTIONS = {
        "create_folder",
        "create_file",
        "copy_file",
        "move_file",
        "rename_file",
        "download_file",
        "open_installer",
        "modify_setting",
    }
    _LOW_ACTIONS = {
        "observe_screen",
        "search_file",
        "open_file",
        "open_folder",
        "scroll",
        "focus_window",
    }

    def classify(self, action: AgentAction) -> AgentRisk:
        text = action.searchable_text()
        if any(term in text for term in self._BLOCKED_TERMS):
            return AgentRisk.BLOCKED
        if any(term in text for term in self._HIGH_TERMS):
            return AgentRisk.HIGH
        if action.action_type == "open_path" and action.payload.get("result_from"):
            return AgentRisk.LOW
        if action.action_type in self._MEDIUM_ACTIONS:
            return AgentRisk.MEDIUM
        if action.action_type in self._LOW_ACTIONS:
            return AgentRisk.LOW
        return action.risk

    def authorize(
        self,
        action: AgentAction,
        *,
        mode: AgentMode,
        confirmed: bool = False,
        strong_confirmed: bool = False,
        pin_verified: bool = False,
    ) -> AgentSafetyResult:
        mode = AgentMode(str(mode))
        risk = self.classify(action)
        if risk == AgentRisk.BLOCKED:
            return AgentSafetyResult(AgentSafetyDecision.BLOCK, risk, "Accion bloqueada por la politica de Agent Mode.")
        if mode in {AgentMode.OBSERVE_ONLY, AgentMode.ASSIST} and action.action_type not in self._LOW_ACTIONS:
            return AgentSafetyResult(
                AgentSafetyDecision.REQUIRE_CONFIRMATION,
                risk,
                "Este modo no ejecuta acciones de control directo.",
            )
        if risk == AgentRisk.HIGH:
            if strong_confirmed and pin_verified:
                return AgentSafetyResult(AgentSafetyDecision.ALLOW, risk, "Confirmacion fuerte y PIN verificados.")
            return AgentSafetyResult(
                AgentSafetyDecision.REQUIRE_STRONG_CONFIRMATION,
                risk,
                "Accion de alto riesgo: requiere confirmacion fuerte y PIN si esta configurado.",
                requires_pin=True,
            )
        if risk == AgentRisk.MEDIUM and not confirmed:
            return AgentSafetyResult(AgentSafetyDecision.REQUIRE_CONFIRMATION, risk, "Accion sensible: requiere Confirm Action.")
        return AgentSafetyResult(AgentSafetyDecision.ALLOW, risk, "Accion permitida por Agent Mode.")


class AgentPlanner:
    def build_plan(self, session: AgentSession) -> list[AgentStep]:
        return list(session.plan)


class AgentRuntime:
    def __init__(self, safety_gate: AgentSafetyGate | None = None) -> None:
        self.safety_gate = safety_gate or AgentSafetyGate()


class AgentModeController:
    def __init__(self, safety_gate: AgentSafetyGate | None = None) -> None:
        self.safety_gate = safety_gate or AgentSafetyGate()
        self.current_mode = AgentMode.GUIDED_CONTROL

    def set_mode(self, mode: AgentMode | str) -> AgentMode:
        self.current_mode = AgentMode(str(mode))
        return self.current_mode

    def create_session(self, goal: str, *, mode: AgentMode | str | None = None) -> AgentSession:
        selected_mode = AgentMode(str(mode)) if mode is not None else self.current_mode
        session = AgentSession(user_goal=goal, mode=selected_mode)
        session.audit.add("session_created", status=session.status.value, detail=goal)
        return session

    def cancel(self, session: AgentSession, *, reason: str = "cancelled by user") -> AgentSession:
        session.status = AgentStatus.CANCELLED
        session.updated_at = _utcnow()
        session.errors.append(reason)
        session.audit.add("session_cancelled", status=session.status.value, detail=reason)
        return session
