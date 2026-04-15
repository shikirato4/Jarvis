from __future__ import annotations

from typing import Any

from .base import (
    ApprovalDecision,
    AutonomyLevel,
    MissionStep,
    MissionStepKind,
    RiskLevel,
)

INTERACTIVE_TARGET_KEYWORDS = (
    "write",
    "click",
    "focus",
    "hotkey",
    "shortcut",
)
HIGH_SENSITIVITY_TARGET_KEYWORDS = (
    "shell",
    "process",
    "delete",
    "remove",
    "system.open",
    "system.launch",
    "unity.create",
    "unity.write",
    "unity.bridge",
    "unity.launch",
    "unity.connect_bridge",
    "unity.disconnect_bridge",
)
SENSITIVE_PAYLOAD_KEYS = ("command", "path", "file_path", "text", "secret", "token", "password", "credential")


def evaluate_step_approval(mission, step: MissionStep) -> dict[str, Any]:
    tags: list[str] = []
    reasons: list[str] = []
    requires_approval = bool(step.requires_approval)

    if step.requires_approval:
        tags.append("planner_marked")
        reasons.append(step.approval_reason or "planner marked this step for approval")

    if step.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        requires_approval = True
        tags.append(f"risk:{step.risk_level.value}")
        reasons.append(f"step risk is {step.risk_level.value}")

    if step.kind in {MissionStepKind.UI, MissionStepKind.VOICE, MissionStepKind.ACTION, MissionStepKind.TOOL}:
        tags.append(f"kind:{step.kind.value}")
        target_sensitivity = _target_sensitivity(step.target)
        if target_sensitivity is not None:
            reasons.append(f"sensitive target '{step.target}'")
            tags.append(f"sensitive_target:{target_sensitivity}")
            if target_sensitivity == "high":
                requires_approval = True
        if step.target.startswith("system."):
            tags.append("system_runtime")
            reasons.append("system runtime side effect")
            if step.target in {"system.open_target", "system.launch_application", "system.open_path", "system.reveal_target"}:
                requires_approval = True
        if step.target.startswith("unity."):
            tags.append("unity_runtime")
            reasons.append("unity runtime side effect")
            if step.target in {
                "unity.create_project",
                "unity.write_script",
                "unity.open_project",
                "unity.launch_project",
                "unity.editor_operation",
                "unity.editor_command",
                "unity.bridge_command",
                "unity.connect_bridge",
                "unity.disconnect_bridge",
            }:
                requires_approval = True

    sensitivity = _inspect_payload(step.payload)
    if sensitivity["requires_approval"]:
        requires_approval = True
        reasons.extend(sensitivity["reasons"])
        tags.extend(sensitivity["tags"])

    if mission.policy.level in {AutonomyLevel.MANUAL, AutonomyLevel.ASSISTED}:
        requires_approval = True
        tags.append(f"policy:{mission.policy.level.value}")
        reasons.append(f"mission level '{mission.policy.level.value}' requires operator confirmation")

    if step.kind == MissionStepKind.UI and mission.policy.require_confirmation_for_ui:
        requires_approval = True
        tags.append("policy:ui_confirmation")
        reasons.append("policy requires confirmation for ui steps")

    if step.kind == MissionStepKind.VOICE and mission.policy.require_confirmation_for_voice:
        requires_approval = True
        tags.append("policy:voice_confirmation")
        reasons.append("policy requires confirmation for voice steps")

    approval_reason = "; ".join(dict.fromkeys(reason.strip() for reason in reasons if reason.strip())) or None
    recommended_action = ApprovalDecision.APPROVE.value if not requires_approval else ApprovalDecision.PAUSE.value
    return {
        "requires_approval": requires_approval,
        "approval_reason": approval_reason,
        "approval_tags": list(dict.fromkeys(tags)),
        "recommended_action": recommended_action,
    }


def _target_sensitivity(target: str) -> str | None:
    lowered = target.lower()
    if any(keyword in lowered for keyword in HIGH_SENSITIVITY_TARGET_KEYWORDS):
        return "high"
    if any(keyword in lowered for keyword in INTERACTIVE_TARGET_KEYWORDS):
        return "interactive"
    return None


def _inspect_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    reasons: list[str] = []
    requires_approval = False

    for key, value in payload.items():
        normalized = key.lower()
        if normalized not in SENSITIVE_PAYLOAD_KEYS:
            continue
        text = str(value)
        if normalized == "text" and len(text) > 120:
            requires_approval = True
            tags.append("payload:long_text")
            reasons.append("payload includes long text to write")
        elif normalized in {"command", "path", "file_path"} and text:
            requires_approval = True
            tags.append(f"payload:{normalized}")
            reasons.append(f"payload includes {normalized}")
        elif normalized in {"secret", "token", "password", "credential"} and text:
            requires_approval = True
            tags.append("payload:sensitive_data")
            reasons.append("payload includes sensitive data")

    return {"requires_approval": requires_approval, "reasons": reasons, "tags": tags}
