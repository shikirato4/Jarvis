from __future__ import annotations

from collections import Counter

from .base import AutonomousMission, MissionStatus, RetryDecision, StopReason


def detect_loop(mission: AutonomousMission, *, repetition_limit: int = 3) -> bool:
    signatures = [f"{item.step_id}:{item.message}" for item in mission.step_results[-repetition_limit:]]
    if len(signatures) < repetition_limit:
        return False
    counts = Counter(signatures)
    return any(value >= repetition_limit for value in counts.values())


def detect_no_progress(mission: AutonomousMission, *, recent_window: int = 3) -> bool:
    recent = mission.step_results[-recent_window:]
    if len(recent) < recent_window:
        return False
    return all(item.status in {"failed", "blocked", "skipped"} for item in recent)


def apply_stop_safeguards(mission: AutonomousMission) -> StopReason | None:
    if detect_loop(mission):
        return StopReason.LOOP_DETECTED
    if detect_no_progress(mission):
        return StopReason.NO_PROGRESS
    return None
