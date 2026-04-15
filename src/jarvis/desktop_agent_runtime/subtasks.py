from __future__ import annotations

from datetime import datetime, timezone

from .models import DesktopAgentPlan, DesktopAgentStep, DesktopAgentSubtask, DesktopMissionStepStatus


def build_subtasks(plan: DesktopAgentPlan) -> list[DesktopAgentSubtask]:
    subtasks: list[DesktopAgentSubtask] = []
    for index, step in enumerate(plan.steps, start=1):
        subtasks.append(
            DesktopAgentSubtask(
                subtask_id=step.step_id,
                label=step.title,
                steps=[step.step_id],
                expected_outcome=step.success_label or step.subgoal or step.title,
                recovery_hints=_collect_recovery_hints(step),
                completion_criteria=_collect_completion_criteria(step),
                notes=[f"subtask_{index}", f"strategy:{plan.strategy}"],
            )
        )
    return subtasks


def mark_subtask_started(subtasks: list[DesktopAgentSubtask], step_id: str) -> tuple[list[DesktopAgentSubtask], DesktopAgentSubtask | None]:
    current: DesktopAgentSubtask | None = None
    now = datetime.now(timezone.utc)
    for subtask in subtasks:
        if step_id in subtask.steps:
            if subtask.status == DesktopMissionStepStatus.PENDING:
                subtask.status = DesktopMissionStepStatus.IN_PROGRESS
                subtask.started_at = now
            current = subtask
    return subtasks, current


def mark_subtask_terminal(subtasks: list[DesktopAgentSubtask], step_id: str, status: DesktopMissionStepStatus, note: str | None = None) -> list[DesktopAgentSubtask]:
    now = datetime.now(timezone.utc)
    for subtask in subtasks:
        if step_id in subtask.steps:
            subtask.status = status
            subtask.completed_at = now
            if note:
                subtask.notes.append(note)
    return subtasks


def _collect_recovery_hints(step: DesktopAgentStep) -> list[str]:
    hints = []
    if step.fallback:
        hints.append(step.fallback)
    hints.extend(step.alternatives)
    return hints


def _collect_completion_criteria(step: DesktopAgentStep) -> list[str]:
    criteria = []
    verification = step.verification.model_dump(mode="json")
    for key, value in verification.items():
        if value not in (None, [], {}, ""):
            criteria.append(f"{key}={value}")
    return criteria or [step.success_label or step.title]
