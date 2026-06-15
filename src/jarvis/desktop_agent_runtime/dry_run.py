from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import DesktopAgentMissionRequest, DesktopPolicyDecision
from .planner import DesktopAgentPlanner
from .policies import DesktopAgentPolicyEngine
from .rollback import RollbackPlanner
from .world_model import DesktopWorldModelBuilder


@dataclass(frozen=True)
class DryRunStep:
    step_id: str
    title: str
    action_type: str
    human_description: str
    risk: str
    target: str
    requires_confirmation: bool
    dry_run_supported: bool
    verify_supported: bool
    rollback: dict[str, Any]
    skill: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "action_type": self.action_type,
            "human_description": self.human_description,
            "risk": self.risk,
            "target": self.target,
            "requires_confirmation": self.requires_confirmation,
            "dry_run_supported": self.dry_run_supported,
            "verify_supported": self.verify_supported,
            "rollback": self.rollback,
            "skill": self.skill,
        }


@dataclass(frozen=True)
class DryRunResult:
    status: str
    goal: str
    strategy: str
    permission_mode: str
    steps: list[DryRunStep] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal,
            "strategy": self.strategy,
            "permission_mode": self.permission_mode,
            "summary": self.summary,
            "steps": [step.to_dict() for step in self.steps],
            "modifies_system": False,
            "executed": False,
        }


class DesktopAgentDryRunPlanner:
    def __init__(self, *, settings, permission_mode: str = "normal") -> None:
        self._settings = settings
        self._permission_mode = permission_mode
        self._planner = DesktopAgentPlanner(settings)
        self._policy = DesktopAgentPolicyEngine(settings)
        self._rollback = RollbackPlanner()

    def plan(self, request: DesktopAgentMissionRequest | dict[str, Any]) -> DryRunResult:
        payload = DesktopAgentMissionRequest.model_validate(request)
        world = DesktopWorldModelBuilder().create(payload)
        plan = self._planner.plan(world)
        steps: list[DryRunStep] = []
        for step in plan.steps:
            policy = self._policy.assess_step(step, permission_mode=self._permission_mode)
            rollback = self._rollback.for_step(step).to_dict()
            steps.append(
                DryRunStep(
                    step_id=step.step_id,
                    title=step.title,
                    action_type=step.action_type.value,
                    human_description=step.action,
                    risk=policy.risk_level.value,
                    target=_step_target(step.payload),
                    requires_confirmation=policy.decision != DesktopPolicyDecision.ALLOW,
                    dry_run_supported=True,
                    verify_supported=True,
                    rollback=rollback,
                    skill=_skill_for_action(step.action_type.value),
                )
            )
        summary = _format_dry_run_summary(payload.goal, steps)
        return DryRunResult(
            status="dry_run",
            goal=payload.goal,
            strategy=plan.strategy,
            permission_mode=self._permission_mode,
            steps=steps,
            summary=summary,
        )


def _step_target(payload: dict[str, Any]) -> str:
    for key in ("path", "destination_path", "application", "target_window", "label", "query"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _skill_for_action(action_type: str) -> str:
    mapping = {
        "observe_screen": "inspect_screen",
        "open_application": "open_application",
        "create_folder": "create_folder",
        "create_file": "create_file",
        "copy_file": "copy_file",
        "move_file": "move_file",
        "rename_file": "rename_file",
        "search_file": "verify_file_exists",
        "open_folder": "open_url",
        "open_path": "open_url",
        "write_text": "controlled_input",
        "hotkey": "controlled_input",
        "click_target": "controlled_input",
    }
    return mapping.get(action_type, "agent_action")


def _format_dry_run_summary(goal: str, steps: list[DryRunStep]) -> str:
    lines = [
        "Esto es lo que haria, sin ejecutarlo todavia:",
        "",
        f"Objetivo: {goal}",
        "",
    ]
    for index, step in enumerate(steps, start=1):
        lines.append(f"{index}. {step.title}")
        lines.append(f"   Accion: {step.human_description}")
        if step.target:
            lines.append(f"   Objetivo: {step.target}")
        lines.append(f"   Riesgo: {step.risk}")
        lines.append(f"   Requiere confirmacion: {'si' if step.requires_confirmation else 'no'}")
        lines.append(f"   Rollback: {step.rollback.get('rollback_description')}")
    lines.append("")
    lines.append("Dry run completado: no movi mouse, no escribi, no descargue y no modifique archivos.")
    return "\n".join(lines)
