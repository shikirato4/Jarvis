from __future__ import annotations

from .models import DesktopAgentProgress, DesktopAgentSubtask, DesktopAgentStepReceipt, DesktopMissionStepStatus


def build_progress(subtasks: list[DesktopAgentSubtask], step_receipts: list[DesktopAgentStepReceipt], failed_steps: list[str]) -> DesktopAgentProgress:
    total_subtasks = len(subtasks)
    completed_subtasks = len([item for item in subtasks if item.status == DesktopMissionStepStatus.COMPLETED])
    total_steps = len({receipt.step_id for receipt in step_receipts} | {step_id for subtask in subtasks for step_id in subtask.steps})
    completed_steps = len({receipt.step_id for receipt in step_receipts if receipt.status.value == "passed"})
    failed_count = len(set(failed_steps))
    percent = 0.0 if total_steps == 0 else round((completed_steps / total_steps) * 100, 2)
    return DesktopAgentProgress(
        total_subtasks=total_subtasks,
        completed_subtasks=completed_subtasks,
        total_steps=total_steps,
        completed_steps=completed_steps,
        failed_steps=failed_count,
        percent_complete=percent,
    )
