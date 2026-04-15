from __future__ import annotations

from uuid import uuid4

from .base import (
    AutonomousMission,
    AutonomyPolicy,
    ExecutionBudget,
    MissionContext,
    MissionGoal,
    MissionRequest,
    MissionState,
    MissionStatus,
)


def build_mission(request: MissionRequest, *, default_policy: AutonomyPolicy, default_budget: ExecutionBudget) -> AutonomousMission:
    mission_id = str(uuid4())
    policy = request.policy or default_policy
    if request.autonomy_level is not None:
        policy = policy.model_copy(update={"level": request.autonomy_level})
    budget = request.budget or default_budget
    goal = MissionGoal(
        title=request.goal[:80],
        objective=request.goal,
        success_criteria=[request.payload.get("success_criteria")] if request.payload.get("success_criteria") else [],
        metadata=request.metadata,
    )
    context = MissionContext(query=request.goal, payload=request.payload, metadata=request.metadata)
    state = MissionState(mission_id=mission_id, status=MissionStatus.PENDING)
    return AutonomousMission(
        mission_id=mission_id,
        goal=goal,
        context=context,
        policy=policy,
        budget=budget,
        state=state,
        metadata=request.metadata,
    )
