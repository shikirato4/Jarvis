from __future__ import annotations

from .base import ReflectionResult, RetryDecision


class MissionReflector:
    def __init__(self, *, logger=None) -> None:
        self._logger = logger

    def reflect(self, mission, step, step_result, verification) -> ReflectionResult:
        if verification.success and verification.goal_satisfied:
            return ReflectionResult(decision=RetryDecision.STOP, message="Goal satisfied after verification.", confidence=verification.confidence)
        if verification.success:
            return ReflectionResult(decision=RetryDecision.SKIP, message="Step verified; continue to next step.", confidence=verification.confidence)
        attempts = sum(1 for item in mission.step_results if item.step_id == step.step_id)
        if attempts <= step.budget.max_retries:
            return ReflectionResult(decision=RetryDecision.RETRY, message="Retry current step.", confidence=0.5)
        if mission.state.replans < mission.budget.max_replans:
            return ReflectionResult(decision=RetryDecision.REPLAN, message="Replan after repeated verification failure.", confidence=0.55, should_replan=True)
        return ReflectionResult(decision=RetryDecision.STOP, message="Stop after repeated verification failure.", confidence=0.7)
