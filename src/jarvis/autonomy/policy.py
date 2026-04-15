from __future__ import annotations

from .base import ActionDecision, AutonomyLevel, AutonomyPolicy, MissionStep, MissionStepKind, RiskLevel


def classify_step_risk(step: MissionStep) -> RiskLevel:
    if step.kind in {MissionStepKind.UI, MissionStepKind.VOICE}:
        if step.target in {"interface.click_mouse", "interface.keyboard_shortcut", "interface.write_text", "voice_runtime.speak"}:
            return RiskLevel.MEDIUM
    if step.kind == MissionStepKind.ACTION and step.target.startswith("operations."):
        return RiskLevel.HIGH
    if step.kind in {MissionStepKind.OBSERVE, MissionStepKind.RETRIEVE, MissionStepKind.VISION, MissionStepKind.VERIFY, MissionStepKind.REFLECT}:
        return RiskLevel.LOW
    return step.risk_level


def decide_action(step: MissionStep, policy: AutonomyPolicy) -> ActionDecision:
    risk = classify_step_risk(step)
    if risk == RiskLevel.CRITICAL and policy.prohibit_critical_steps:
        return ActionDecision.PROHIBIT
    if policy.level == AutonomyLevel.MANUAL:
        return ActionDecision.SUGGEST
    if policy.level == AutonomyLevel.ASSISTED:
        return ActionDecision.SUGGEST
    if policy.level == AutonomyLevel.SEMI_AUTONOMOUS and risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return ActionDecision.REQUIRE_CONFIRMATION
    if policy.level == AutonomyLevel.SUPERVISED_AUTONOMOUS and risk == RiskLevel.CRITICAL:
        return ActionDecision.REQUIRE_CONFIRMATION
    if policy.high_risk_requires_confirmation and risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return ActionDecision.REQUIRE_CONFIRMATION
    if policy.require_confirmation_for_ui and step.kind == MissionStepKind.UI:
        return ActionDecision.REQUIRE_CONFIRMATION
    if policy.require_confirmation_for_voice and step.kind == MissionStepKind.VOICE:
        return ActionDecision.REQUIRE_CONFIRMATION
    return ActionDecision.EXECUTE
