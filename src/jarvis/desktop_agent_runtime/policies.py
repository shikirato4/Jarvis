from __future__ import annotations

from .models import DesktopAgentPolicyResult, DesktopAgentRiskLevel, DesktopAgentStep, DesktopPolicyDecision, DesktopStepActionType


class DesktopAgentPolicyEngine:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._allowed_titles = tuple(item.casefold() for item in settings.ui_allowed_window_titles)
        self._allow_discovered = settings.ui_allow_discovered_applications

    def assess_step(self, step: DesktopAgentStep) -> DesktopAgentPolicyResult:
        trusted_window_tokens = {*self._allowed_titles, "explorer", "explorador"}
        if step.action_type == DesktopStepActionType.OPEN_PATH and not step.payload.get("result_from"):
            return DesktopAgentPolicyResult(
                decision=DesktopPolicyDecision.REQUIRE_CONFIRMATION,
                risk_level=DesktopAgentRiskLevel.MEDIUM,
                reason="Opening arbitrary paths requires confirmation.",
            )
        if step.action_type == DesktopStepActionType.HOTKEY:
            keys = "+".join(str(item).casefold() for item in step.payload.get("keys", ()))
            if keys in {item.casefold() for item in self._settings.ui_hotkey_blocklist}:
                return DesktopAgentPolicyResult(
                    decision=DesktopPolicyDecision.DENY,
                    risk_level=DesktopAgentRiskLevel.HIGH,
                    reason=f"Blocked hotkey: {keys}.",
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
