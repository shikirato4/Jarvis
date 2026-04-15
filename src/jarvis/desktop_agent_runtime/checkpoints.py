from __future__ import annotations

from uuid import uuid4

from .models import DesktopAgentCheckpoint, DesktopAgentMissionReceipt, DesktopWorldState


def build_checkpoint(mission: DesktopAgentMissionReceipt, world: DesktopWorldState) -> DesktopAgentCheckpoint:
    return DesktopAgentCheckpoint(
        checkpoint_id=f"chk-{uuid4().hex[:10]}",
        mission_id=mission.mission_id,
        phase=mission.status,
        current_subtask=mission.current_subtask,
        current_step=world.current_step_id,
        next_step_index=mission.next_step_index,
        observation_summary=world.last_observation_summary,
        active_window_title=world.active_window.title if world.active_window else None,
        target_application=world.target_application,
        target_window_title=world.target_window_title,
        strategy=world.memory.last_strategy,
        attempts=dict(world.memory.recovery_attempts_by_step),
        data={
            "last_error": world.last_error,
            "context_signals": list(world.context_signals),
            "completed_steps": list(mission.completed_steps),
            "failed_steps": list(mission.failed_steps),
            "last_result": dict(world.last_result),
        },
    )
