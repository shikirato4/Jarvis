from __future__ import annotations

import logging
from datetime import datetime, timezone

from jarvis.config import Settings
from jarvis.core.errors import AutonomyRuntimeError, AutonomyValidationError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .base import (
    ActionDecision,
    ApprovalDecision,
    AutonomousMission,
    AutonomyLevel,
    AutonomyPolicy,
    ExecutionBudget,
    MissionApprovalRequest,
    MissionControlActionRequest,
    MissionControlView,
    MissionPlan,
    MissionPlanRequest,
    MissionReceipt,
    MissionRequest,
    MissionState,
    MissionStatus,
    MissionStatusView,
    MissionStep,
    MissionStepResult,
    MissionStepStatus,
    RetryDecision,
    StopReason,
    VerificationRequest,
)
from .budget import effective_budget, evaluate_budget_stop, register_risk
from .mission import build_mission
from .planner import MissionPlanner
from .policy import classify_step_risk, decide_action
from .reflection import MissionReflector
from .safeguards import apply_stop_safeguards
from .state import MissionStateManager
from .verifier import MissionVerifier


class AutonomyService:
    service_name = "autonomy"

    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        event_bus: EventBus,
        planner: MissionPlanner,
        executor,
        verifier: MissionVerifier,
        reflector: MissionReflector,
        state_manager: MissionStateManager,
        mission_control,
        persistence,
        *,
        logger: logging.Logger | None = None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._event_bus = event_bus
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._reflector = reflector
        self._state = state_manager
        self._control = mission_control
        self._persistence = persistence
        self._logger = logger or logging.getLogger("jarvis.autonomy")
        self._operations = operation_registry
        self._default_budget = ExecutionBudget(
            max_steps=settings.autonomy_max_steps,
            max_duration_seconds=settings.autonomy_max_duration_seconds,
            max_replans=settings.autonomy_max_replans,
            max_retries_per_step=settings.autonomy_max_retries_per_step,
        )
        self._default_policy = AutonomyPolicy(
            level=AutonomyLevel(settings.autonomy_default_level),
            high_risk_requires_confirmation=settings.autonomy_high_risk_requires_confirmation,
            stop_on_low_confidence=settings.autonomy_stop_on_low_confidence,
        )

    def status(self) -> dict[str, object]:
        active = self._state.active()
        return MissionStatusView(
            active_mission_id=active.mission_id if active else None,
            active_level=active.policy.level.value if active else None,
            missions=[
                {
                    "mission_id": mission.mission_id,
                    "status": mission.state.status.value,
                    "goal": mission.goal.objective,
                    "autonomy_level": mission.policy.level.value,
                }
                for mission in self._state.list()[:20]
            ],
            default_level=self._default_policy.level.value,
            enabled=self._settings.autonomy_enabled,
        ).model_dump(mode="json")

    def plan_mission(self, request: MissionPlanRequest | MissionRequest | dict[str, object]) -> MissionPlan:
        mission_request = self._coerce_request(request)
        mission = build_mission(mission_request, default_policy=self._default_policy, default_budget=effective_budget(self._default_budget, mission_request.budget))
        plan = self._planner.plan(mission)
        self._event_bus.publish(
            "autonomy.step_planned",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.plan",
                "status": "planned",
                "autonomy_level": mission.policy.level.value,
                "goal": mission.goal.objective,
                "data": {"step_count": len(plan.steps), "strategy_name": plan.strategy_name},
            },
        )
        return plan

    def start_mission(self, request: MissionRequest | dict[str, object]) -> MissionReceipt:
        if not self._settings.autonomy_enabled:
            raise AutonomyRuntimeError("autonomy is disabled by configuration")
        active_running = sum(1 for mission in self._state.list() if mission.state.status in {MissionStatus.RUNNING, MissionStatus.PLANNING, MissionStatus.PENDING})
        if active_running >= self._settings.autonomy_max_concurrent_missions:
            raise AutonomyRuntimeError(
                "autonomy mission limit reached",
                details={"active_running": active_running, "limit": self._settings.autonomy_max_concurrent_missions},
                recoverable=True,
            )
        mission_request = self._coerce_request(request)
        mission = build_mission(
            mission_request,
            default_policy=self._default_policy,
            default_budget=effective_budget(self._default_budget, mission_request.budget),
        )
        mission.state = mission.state.model_copy(update={"status": MissionStatus.PLANNING})
        mission.plan = self._planner.plan(mission)
        mission.state = mission.state.model_copy(update={"status": MissionStatus.RUNNING})
        self._save_mission(mission)
        self._persistence.append_event(mission, "started", payload={"goal": mission.goal.objective})
        self._persistence.append_event(
            mission,
            "planned",
            payload={"step_count": len(mission.plan.steps) if mission.plan else 0, "strategy_name": mission.plan.strategy_name if mission.plan else None},
        )
        self._event_bus.publish(
            "autonomy.started",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.start",
                "autonomy_level": mission.policy.level.value,
                "goal": mission.goal.objective,
                "data": {"step_count": len(mission.plan.steps) if mission.plan else 0},
            },
        )
        return self._receipt(mission, message="mission started")

    def step_mission(self, mission_id: str) -> MissionReceipt:
        mission = self._require_mission(mission_id)
        if mission.plan is None:
            mission.plan = self._planner.plan(mission)
            self._save_mission(mission)
        if mission.state.status in {MissionStatus.COMPLETED, MissionStatus.CANCELLED, MissionStatus.FAILED, MissionStatus.STOPPED}:
            return self._receipt(mission, message="mission already finished")
        if mission.state.status == MissionStatus.PAUSED:
            return self._receipt(mission, message="mission is paused")
        if mission.state.status in {MissionStatus.WAITING_CONFIRMATION, MissionStatus.AWAITING_REVIEW}:
            return self._receipt(mission, message="mission is waiting for approval")
        stop_reason = evaluate_budget_stop(mission) or apply_stop_safeguards(mission)
        if stop_reason is not None:
            mission = self._stop_mission(mission, stop_reason, "mission stopped by safeguard or budget")
            return self._receipt(mission, message="mission stopped")
        step = self._next_step(mission)
        if step is None:
            verification = self._verifier.verify_mission(mission)
            if verification.goal_satisfied:
                mission = self._complete_mission(mission, StopReason.GOAL_SATISFIED, verification.message)
            else:
                mission = self._stop_mission(mission, StopReason.NO_PROGRESS, "no executable steps remain")
            return self._receipt(mission, verification=verification, message=mission.state.status.value)

        decision = decide_action(step, mission.policy)
        if decision in {ActionDecision.SUGGEST, ActionDecision.REQUIRE_CONFIRMATION}:
            mission = self._control.request_step_approval(
                mission,
                step,
                reason=f"policy decision: {decision.value}",
                status=MissionStatus.WAITING_CONFIRMATION,
            )
            self._event_bus.publish(
                "autonomy.stopped",
                {
                    "mission_id": mission.mission_id,
                    "operation_name": "autonomy.confirmation_required",
                    "status": mission.state.status.value,
                    "autonomy_level": mission.policy.level.value,
                    "step_id": step.step_id,
                    "goal": mission.goal.objective,
                    "stop_reason": StopReason.USER_CONFIRMATION_REQUIRED.value,
                    "data": {"decision": decision.value, "step_title": step.title},
                },
            )
            return self._receipt(mission, current_step=step, message="confirmation required")
        if decision == ActionDecision.PROHIBIT:
            mission = self._stop_mission(mission, StopReason.POLICY_STOP, "policy prohibited the next step")
            return self._receipt(mission, current_step=step, message="policy stop")

        step.risk_level = classify_step_risk(step)
        gate = self._control.should_gate_step(mission, step)
        if gate["requires_approval"]:
            mission = self._control.request_step_approval(mission, step, reason=str(gate["approval_reason"] or "approval required"))
            self._event_bus.publish(
                "autonomy.stopped",
                {
                    "mission_id": mission.mission_id,
                    "operation_name": "autonomy.approval_required",
                    "status": mission.state.status.value,
                    "autonomy_level": mission.policy.level.value,
                    "step_id": step.step_id,
                    "goal": mission.goal.objective,
                    "stop_reason": StopReason.USER_CONFIRMATION_REQUIRED.value,
                    "data": gate,
                },
            )
            return self._receipt(mission, current_step=step, message="approval required")
        mission.state = register_risk(mission.state.model_copy(update={"status": MissionStatus.RUNNING, "active_step_id": step.step_id}), step.risk_level)
        step.status = MissionStepStatus.RUNNING
        self._save_mission(mission)
        self._persistence.append_event(mission, "step_started", step_id=step.step_id, payload={"step_title": step.title, "target": step.target})
        result = self._executor.execute(mission, step)
        verification = self._verifier.verify_step(
            VerificationRequest(mission_id=mission.mission_id, step=step, result_data=result.data, observation=self._executor.observe(mission))
        )
        self._verifier.update_summary(mission, verification)
        result.status = MissionStepStatus.VERIFIED if verification.success else MissionStepStatus.FAILED
        mission.step_results.append(result)
        mission.verification_history.append(verification)
        mission.state = mission.state.model_copy(
            update={
                "executed_steps": mission.state.executed_steps + 1,
                "step_index": mission.state.step_index + 1,
                "verification_failures": mission.state.verification_failures + (0 if verification.success else 1),
                "failures": mission.state.failures + (0 if verification.success else 1),
                "waiting_for_confirmation": False,
                "pending_approval_step_id": None,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._save_mission(mission)
        self._persistence.append_event(
            mission,
            "step_verified",
            step_id=step.step_id,
            payload={"verification_success": verification.success, "failure_code": verification.failure_code, "goal_satisfied": verification.goal_satisfied},
        )
        self._event_bus.publish(
            "autonomy.step_executed",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.step",
                "status": result.status.value,
                "autonomy_level": mission.policy.level.value,
                "step_id": step.step_id,
                "goal": mission.goal.objective,
                "data": {"step_title": step.title, "verification_success": verification.success},
            },
        )
        self._event_bus.publish(
            "autonomy.verified",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.verify",
                "status": "verified" if verification.success else "verification_failed",
                "autonomy_level": mission.policy.level.value,
                "step_id": step.step_id,
                "goal": mission.goal.objective,
                "data": verification.model_dump(mode="json"),
            },
        )
        if verification.success and verification.goal_satisfied:
            mission = self._complete_mission(mission, StopReason.GOAL_SATISFIED, verification.message)
            return self._receipt(mission, current_step=step, verification=verification, message="goal satisfied")
        reflection = self._reflector.reflect(mission, step, result, verification)
        mission.reflection_history.append(reflection)
        if reflection.decision == RetryDecision.REPLAN and reflection.should_replan:
            mission.plan = self._planner.replan(mission, reason=reflection.message)
            mission.state = mission.state.model_copy(update={"replans": mission.state.replans + 1, "status": MissionStatus.RUNNING, "updated_at": datetime.now(timezone.utc)})
            self._event_bus.publish(
                "autonomy.replanned",
                {
                    "mission_id": mission.mission_id,
                    "operation_name": "autonomy.replan",
                    "status": "replanned",
                    "autonomy_level": mission.policy.level.value,
                    "step_id": step.step_id,
                    "goal": mission.goal.objective,
                    "data": {"reason": reflection.message, "step_count": len(mission.plan.steps)},
                },
            )
        elif reflection.decision == RetryDecision.STOP:
            mission = self._stop_mission(mission, StopReason.VERIFICATION_FAILED, reflection.message)
        self._save_mission(mission)
        return self._receipt(mission, current_step=step, verification=verification, reflection=reflection, message=reflection.message)

    def run_mission(self, mission_id: str) -> MissionReceipt:
        mission = self._require_mission(mission_id)
        guard = 0
        last_receipt = self._receipt(mission, message="mission loaded")
        while mission.state.status in {MissionStatus.RUNNING, MissionStatus.PLANNING, MissionStatus.PENDING} and guard < mission.budget.max_steps + 2:
            last_receipt = self.step_mission(mission_id)
            mission = self._require_mission(mission_id)
            guard += 1
            if mission.state.status in {
                MissionStatus.WAITING_CONFIRMATION,
                MissionStatus.AWAITING_REVIEW,
                MissionStatus.PAUSED,
                MissionStatus.COMPLETED,
                MissionStatus.FAILED,
                MissionStatus.STOPPED,
                MissionStatus.CANCELLED,
            }:
                break
        if guard >= mission.budget.max_steps + 2 and mission.state.status == MissionStatus.RUNNING:
            mission = self._stop_mission(mission, StopReason.LOOP_DETECTED, "run loop guard reached")
            last_receipt = self._receipt(mission, message="loop guard reached")
        return last_receipt

    def stop_mission(self, mission_id: str) -> MissionReceipt:
        mission = self._require_mission(mission_id)
        if self._operations is not None:
            self._operations.cancel_by_metadata("mission_id", mission_id, reason="mission stopped")
        mission = self._stop_mission(mission, StopReason.CANCELLED, "mission stopped by request")
        return self._receipt(mission, message="mission stopped")

    def approve_step(self, request: MissionApprovalRequest | dict[str, object]) -> MissionReceipt:
        approval = MissionApprovalRequest.model_validate(request)
        mission = self._require_mission(approval.mission_id)
        mission = self._control.apply_approval(mission, approval)
        return self._receipt(mission, message=f"approval decision applied: {approval.decision.value}")

    def reject_step(self, request: MissionApprovalRequest | dict[str, object]) -> MissionReceipt:
        approval = MissionApprovalRequest.model_validate(request)
        approval = approval.model_copy(update={"decision": ApprovalDecision.REJECT})
        mission = self._require_mission(approval.mission_id)
        mission = self._control.apply_approval(mission, approval)
        return self._receipt(mission, message="step rejected")

    def pause_mission(self, request: MissionControlActionRequest | dict[str, object]) -> MissionReceipt:
        action = MissionControlActionRequest.model_validate(request)
        mission = self._require_mission(action.mission_id)
        if self._operations is not None:
            self._operations.cancel_by_metadata("mission_id", mission.mission_id, reason="mission paused")
        mission = self._control.pause_mission(mission, action)
        return self._receipt(mission, message="mission paused")

    def resume_mission(self, request: MissionControlActionRequest | dict[str, object]) -> MissionReceipt:
        action = MissionControlActionRequest.model_validate(request)
        mission = self._require_mission(action.mission_id)
        mission = self._control.resume_mission(mission, action)
        return self._receipt(mission, message="mission resumed")

    def retry_step(self, request: MissionControlActionRequest | dict[str, object]) -> MissionReceipt:
        action = MissionControlActionRequest.model_validate(request)
        mission = self._require_mission(action.mission_id)
        mission = self._control.retry_step(mission, action)
        return self._receipt(mission, message="mission step marked for retry")

    def skip_step(self, request: MissionControlActionRequest | dict[str, object]) -> MissionReceipt:
        action = MissionControlActionRequest.model_validate(request)
        mission = self._require_mission(action.mission_id)
        mission = self._control.skip_step(mission, action)
        return self._receipt(mission, message="mission step skipped")

    def mission_control_view(self, mission_id: str) -> MissionControlView:
        mission = self._require_mission(mission_id)
        return self._control.build_view(mission)

    def inspect_mission(self, mission_id: str) -> MissionReceipt:
        mission = self._require_mission(mission_id)
        return self._receipt(mission, message="mission snapshot")

    def list_missions(self) -> list[dict[str, object]]:
        return [
            {
                "mission_id": mission.mission_id,
                "goal": mission.goal.objective,
                "status": mission.state.status.value,
                "autonomy_level": mission.policy.level.value,
                "executed_steps": mission.state.executed_steps,
                "replans": mission.state.replans,
                "stop_reason": mission.state.stop_reason.value if mission.state.stop_reason else None,
            }
            for mission in self._state.list()
        ]

    def _coerce_request(self, request) -> MissionRequest:
        if isinstance(request, MissionPlanRequest):
            return MissionRequest(goal=request.goal, payload=request.payload, autonomy_level=request.autonomy_level, metadata=request.metadata)
        if isinstance(request, MissionRequest):
            return request
        if isinstance(request, dict):
            return MissionRequest.model_validate(request)
        raise AutonomyValidationError("invalid mission request")

    def _require_mission(self, mission_id: str) -> AutonomousMission:
        mission = self._state.get(mission_id)
        if mission is None:
            raise AutonomyValidationError("mission not found", details={"mission_id": mission_id})
        return mission

    def _next_step(self, mission: AutonomousMission) -> MissionStep | None:
        if mission.plan is None:
            return None
        verified_ids = {result.step_id for result in mission.step_results if result.status == MissionStepStatus.VERIFIED}
        failed_counts: dict[str, int] = {}
        for result in mission.step_results:
            if result.status == MissionStepStatus.FAILED:
                failed_counts[result.step_id] = failed_counts.get(result.step_id, 0) + 1
        for step in mission.plan.steps:
            if step.step_id in verified_ids:
                continue
            if step.depends_on and not all(item in verified_ids for item in step.depends_on):
                continue
            attempts = failed_counts.get(step.step_id, 0)
            if attempts > step.budget.max_retries:
                continue
            return step
        return None

    def _complete_mission(self, mission: AutonomousMission, reason: StopReason, message: str) -> AutonomousMission:
        mission.state = mission.state.model_copy(update={"status": MissionStatus.COMPLETED, "stop_reason": reason, "updated_at": datetime.now(timezone.utc)})
        self._save_mission(mission)
        self._persistence.append_event(
            mission,
            "stopped",
            step_id=mission.state.active_step_id,
            payload={"reason": reason.value, "message": message, "status": MissionStatus.COMPLETED.value},
        )
        self._event_bus.publish(
            "autonomy.stopped",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.completed",
                "status": MissionStatus.COMPLETED.value,
                "autonomy_level": mission.policy.level.value,
                "goal": mission.goal.objective,
                "stop_reason": reason.value,
                "data": {"message": message},
            },
        )
        return mission

    def _stop_mission(self, mission: AutonomousMission, reason: StopReason, message: str) -> AutonomousMission:
        status = MissionStatus.CANCELLED if reason == StopReason.CANCELLED else MissionStatus.STOPPED
        mission.state = mission.state.model_copy(update={"status": status, "stop_reason": reason, "updated_at": datetime.now(timezone.utc), "last_error": message})
        self._save_mission(mission)
        self._persistence.append_event(
            mission,
            "stopped",
            step_id=mission.state.active_step_id,
            payload={"reason": reason.value, "message": message},
        )
        self._event_bus.publish(
            "autonomy.stopped" if status != MissionStatus.FAILED else "autonomy.failed",
            {
                "mission_id": mission.mission_id,
                "operation_name": "autonomy.stop",
                "status": status.value,
                "autonomy_level": mission.policy.level.value,
                "goal": mission.goal.objective,
                "stop_reason": reason.value,
                "data": {"message": message},
            },
        )
        return mission

    def _receipt(
        self,
        mission: AutonomousMission,
        *,
        message: str,
        current_step: MissionStep | None = None,
        verification=None,
        reflection=None,
    ) -> MissionReceipt:
        return MissionReceipt(
            mission_id=mission.mission_id,
            status=mission.state.status,
            goal=mission.goal,
            state=mission.state,
            current_step=current_step,
            plan_summary=mission.plan.summary if mission.plan else None,
            verification=verification,
            reflection=reflection,
            recent_results=mission.step_results[-5:],
            message=message,
        )

    def _save_mission(self, mission: AutonomousMission) -> AutonomousMission:
        self._state.save(mission)
        self._persistence.save_mission(mission)
        return mission
