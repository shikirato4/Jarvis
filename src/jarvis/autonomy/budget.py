from __future__ import annotations

from datetime import datetime, timezone

from .base import AutonomousMission, ExecutionBudget, MissionState, RiskLevel, StopReason


RISK_WEIGHTS = {
    RiskLevel.LOW: 0.25,
    RiskLevel.MEDIUM: 1.0,
    RiskLevel.HIGH: 2.5,
    RiskLevel.CRITICAL: 5.0,
}


def effective_budget(default_budget: ExecutionBudget, override: ExecutionBudget | None) -> ExecutionBudget:
    if override is None:
        return default_budget
    return ExecutionBudget.model_validate({**default_budget.model_dump(mode="json"), **override.model_dump(mode="json")})


def register_risk(state: MissionState, risk: RiskLevel) -> MissionState:
    payload = {
        "accumulated_risk": state.accumulated_risk + RISK_WEIGHTS[risk],
        "high_risk_steps": state.high_risk_steps + (1 if risk in {RiskLevel.HIGH, RiskLevel.CRITICAL} else 0),
        "updated_at": datetime.now(timezone.utc),
    }
    return state.model_copy(update=payload)


def evaluate_budget_stop(mission: AutonomousMission) -> StopReason | None:
    state = mission.state
    budget = mission.budget
    elapsed_seconds = (datetime.now(timezone.utc) - mission.created_at).total_seconds()
    if state.executed_steps >= budget.max_steps:
        return StopReason.BUDGET_EXHAUSTED
    if elapsed_seconds >= budget.max_duration_seconds:
        return StopReason.BUDGET_EXHAUSTED
    if state.replans >= budget.max_replans:
        return StopReason.BUDGET_EXHAUSTED
    if state.failures >= budget.max_failures:
        return StopReason.BUDGET_EXHAUSTED
    if state.high_risk_steps > budget.max_high_risk_steps:
        return StopReason.RISK_LIMIT
    if state.observation_cycles > budget.max_observation_cycles:
        return StopReason.BUDGET_EXHAUSTED
    if state.verification_failures > budget.max_verification_failures:
        return StopReason.VERIFICATION_FAILED
    return None
