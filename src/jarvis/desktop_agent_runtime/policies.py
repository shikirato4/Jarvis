from __future__ import annotations

from .agent_mode import AgentAction, AgentRisk, AgentSafetyDecision, AgentSafetyGate
from .models import DesktopAgentPolicyResult, DesktopAgentRiskLevel, DesktopAgentStep, DesktopPolicyDecision, DesktopStepActionType


class DesktopAgentPolicyEngine:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._allowed_titles = tuple(item.casefold() for item in settings.ui_allowed_window_titles)
        self._allow_discovered = settings.ui_allow_discovered_applications
        self._safety_gate = AgentSafetyGate()

    def assess_step(self, step: DesktopAgentStep) -> DesktopAgentPolicyResult:
        if step.action_type == DesktopStepActionType.HOTKEY:
            keys = "+".join(str(item).casefold() for item in step.payload.get("keys", ()))
            if keys in {item.casefold() for item in self._settings.ui_hotkey_blocklist}:
                return DesktopAgentPolicyResult(
                    decision=DesktopPolicyDecision.DENY,
                    risk_level=DesktopAgentRiskLevel.HIGH,
                    reason=f"Blocked hotkey: {keys}.",
                )
        gate_result = self._safety_gate.authorize(
            AgentAction(
                action_type=step.action_type.value,
                title=step.title,
                description=step.action,
                payload=step.payload,
                risk=_agent_risk_from_desktop(step.risk_level),
            ),
            mode="guided_control",
        )
        if gate_result.decision == AgentSafetyDecision.BLOCK:
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.DENY,
                risk_level=DesktopAgentRiskLevel.HIGH,
                reason=gate_result.reason,
                metadata={"agent_mode_risk": gate_result.risk.value},
            )
        if gate_result.decision == AgentSafetyDecision.REQUIRE_STRONG_CONFIRMATION:
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.REQUIRE_CONFIRMATION,
                risk_level=DesktopAgentRiskLevel.HIGH,
                reason=gate_result.reason,
                metadata={"agent_mode_risk": gate_result.risk.value, "requires_pin": gate_result.requires_pin},
            )
        if gate_result.decision == AgentSafetyDecision.REQUIRE_CONFIRMATION:
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.REQUIRE_CONFIRMATION,
                risk_level=_desktop_risk_from_agent(gate_result.risk),
                reason=gate_result.reason,
                metadata={"agent_mode_risk": gate_result.risk.value},
            )

        trusted_window_tokens = {*self._allowed_titles, "explorer", "explorador"}
        if step.action_type == DesktopStepActionType.OPEN_PATH and not step.payload.get("result_from"):
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.REQUIRE_CONFIRMATION,
                risk_level=DesktopAgentRiskLevel.MEDIUM,
                reason="Opening arbitrary paths requires confirmation.",
            )
        application = str(step.payload.get("application") or "").casefold()
        if step.action_type == DesktopStepActionType.OPEN_APPLICATION and application:
            if not any(item in application for item in trusted_window_tokens):
                decision = DesktopPolicyDecision.ALLOW if self._allow_discovered else DesktopPolicyDecision.REQUIRE_CONFIRMATION
                return DesktopAgentPolicyResult(
                    decision=decision,
                    risk_level=DesktopAgentRiskLevel.MEDIUM,
                    reason=f"Application '{application}' is outside the trusted app allowlist.",
                    metadata={"trusted_allowlist": list(self._allowed_titles)},
                )
        target_window = str(step.payload.get("target_window") or step.payload.get("application") or "").casefold()
        if target_window and not any(item in target_window for item in trusted_window_tokens):
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.REQUIRE_CONFIRMATION,
                risk_level=DesktopAgentRiskLevel.MEDIUM,
                reason=f"Target window '{target_window}' is outside the trusted desktop policy.",
            )
        return DesktopAgentPolicyResult(
            decision=DesktopPolicyDecision.ALLOW,
            risk_level=step.risk_level,
            reason="Step allowed by desktop agent policy.",
        )


def _agent_risk_from_desktop(risk: DesktopAgentRiskLevel) -> AgentRisk:
    if risk == DesktopAgentRiskLevel.HIGH:
        return AgentRisk.HIGH
    if risk == DesktopAgentRiskLevel.MEDIUM:
        return AgentRisk.MEDIUM
    if risk == DesktopAgentRiskLevel.CRITICAL:
        return AgentRisk.BLOCKED
    return AgentRisk.LOW


def _desktop_risk_from_agent(risk: AgentRisk) -> DesktopAgentRiskLevel:
    if risk == AgentRisk.HIGH:
        return DesktopAgentRiskLevel.HIGH
    if risk == AgentRisk.MEDIUM:
        return DesktopAgentRiskLevel.MEDIUM
    if risk == AgentRisk.BLOCKED:
        return DesktopAgentRiskLevel.CRITICAL
    return DesktopAgentRiskLevel.LOW
