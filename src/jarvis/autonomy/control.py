from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from jarvis.core.errors import AutonomyValidationError

from .approval_policy import evaluate_step_approval
from .base import (
    ApprovalDecision,
    AutonomousMission,
    MissionApprovalRecord,
    MissionApprovalRequest,
    MissionControlActionRequest,
    MissionControlView,
    MissionState,
    MissionStatus,
    MissionStep,
    MissionStepResult,
    MissionStepStatus,
    StopReason,
)


class MissionControlService:
    def __init__(self, persistence, state_manager, *, logger=None) -> None:
        self._persistence = persistence
        self._state = state_manager
        self._logger = logger

    def should_gate_step(self, mission: AutonomousMission, step: MissionStep) -> dict[str, object]:
        decision = evaluate_step_approval(mission, step)
        step.requires_approval = bool(decision["requires_approval"])
        step.approval_reason = decision["approval_reason"]
        step.approval_tags = list(decision["approval_tags"])
        return decision

    def request_step_approval(
        self,
        mission: AutonomousMission,
        step: MissionStep,
        *,
        reason: str | None = None,
        status: MissionStatus = MissionStatus.AWAITING_REVIEW,
    ) -> AutonomousMission:
        now = datetime.now(timezone.utc)
        mission.state = mission.state.model_copy(
            update={
                "status": status,
                "waiting_for_confirmation": True,
                "pending_approval_step_id": step.step_id,
                "active_step_id": step.step_id,
                "stop_reason": StopReason.USER_CONFIRMATION_REQUIRED,
                "updated_at": now,
            }
        )
        mission.updated_at = now
        self._state.save(mission)
        self._persistence.save_mission(mission)
        self._persistence.append_event(
            mission,
            "approval_requested",
            step_id=step.step_id,
            payload={
                "reason": reason or step.approval_reason or "operator approval required",
                "approval_tags": step.approval_tags,
            },
        )
        return mission

    def apply_approval(self, mission: AutonomousMission, request: MissionApprovalRequest) -> AutonomousMission:
        step_id = request.step_id or mission.state.pending_approval_step_id
        if step_id is None:
            raise AutonomyValidationError("no approval is pending for mission", details={"mission_id": mission.mission_id})
        approval = MissionApprovalRecord(
            mission_id=mission.mission_id,
            step_id=step_id,
            decision=request.decision,
            reason=request.reason,
            actor=request.actor,
            metadata=request.metadata,
        )
        self._persistence.record_approval(mission, approval)
        if request.decision == ApprovalDecision.APPROVE:
            return self._mark_approved(mission, approval)
        if request.decision == ApprovalDecision.REJECT:
            return self._mark_rejected(mission, approval)
        if request.decision == ApprovalDecision.SKIP:
            return self.skip_step(
                mission,
                MissionControlActionRequest(
                    mission_id=mission.mission_id,
                    step_id=step_id,
                    reason=request.reason,
                    actor=request.actor,
                    metadata=request.metadata,
                ),
            )
        if request.decision == ApprovalDecision.PAUSE:
            return self.pause_mission(
                mission,
                MissionControlActionRequest(
                    mission_id=mission.mission_id,
                    step_id=step_id,
                    reason=request.reason,
                    actor=request.actor,
                    metadata=request.metadata,
                ),
            )
        return self.cancel_mission(
            mission,
            MissionControlActionRequest(
                mission_id=mission.mission_id,
                step_id=step_id,
                reason=request.reason,
                actor=request.actor,
                metadata=request.metadata,
            ),
        )

    def pause_mission(self, mission: AutonomousMission, request: MissionControlActionRequest) -> AutonomousMission:
        return self._update_state(
            mission,
            MissionStatus.PAUSED,
            paused=True,
            event_type="paused",
            payload={"reason": request.reason, "actor": request.actor, "step_id": request.step_id},
            step_id=request.step_id or mission.state.active_step_id,
            last_error=request.reason,
        )

    def resume_mission(self, mission: AutonomousMission, request: MissionControlActionRequest) -> AutonomousMission:
        if mission.state.pending_approval_step_id:
            raise AutonomyValidationError(
                "mission cannot resume while approval is pending",
                details={"mission_id": mission.mission_id, "step_id": mission.state.pending_approval_step_id},
            )
        updated = self._update_state(
            mission,
            MissionStatus.RUNNING,
            paused=False,
            event_type="resumed",
            payload={"reason": request.reason, "actor": request.actor},
            step_id=mission.state.active_step_id,
            clear_stop_reason=True,
        )
        updated.state = updated.state.model_copy(update={"resume_token": str(uuid4())})
        self._state.save(updated)
        self._persistence.save_mission(updated)
        return updated

    def cancel_mission(self, mission: AutonomousMission, request: MissionControlActionRequest) -> AutonomousMission:
        return self._update_state(
            mission,
            MissionStatus.CANCELLED,
            paused=False,
            event_type="stopped",
            payload={"reason": request.reason, "actor": request.actor, "decision": ApprovalDecision.CANCEL.value},
            step_id=request.step_id or mission.state.active_step_id,
            stop_reason=StopReason.CANCELLED,
            last_error=request.reason or "mission cancelled by operator",
        )

    def retry_step(self, mission: AutonomousMission, request: MissionControlActionRequest) -> AutonomousMission:
        step = self._resolve_step(mission, request.step_id or mission.state.active_step_id)
        mission.step_results = [
            result for result in mission.step_results if not (result.step_id == step.step_id and result.status == MissionStepStatus.FAILED)
        ]
        step.status = MissionStepStatus.PENDING
        updated = self._update_state(
            mission,
            MissionStatus.RUNNING,
            paused=False,
            event_type="step_retried",
            payload={"reason": request.reason, "actor": request.actor},
            step_id=step.step_id,
            clear_stop_reason=True,
        )
        updated.state = updated.state.model_copy(update={"active_step_id": step.step_id})
        self._state.save(updated)
        self._persistence.save_mission(updated)
        return updated

    def skip_step(self, mission: AutonomousMission, request: MissionControlActionRequest) -> AutonomousMission:
        step = self._resolve_step(mission, request.step_id or mission.state.pending_approval_step_id or mission.state.active_step_id)
        step.status = MissionStepStatus.SKIPPED
        mission.step_results.append(
            MissionStepResult(
                mission_id=mission.mission_id,
                step_id=step.step_id,
                status=MissionStepStatus.SKIPPED,
                message=request.reason or "step skipped by mission control",
                data={"actor": request.actor, "source": "mission_control"},
            )
        )
        return self._update_state(
            mission,
            MissionStatus.RUNNING,
            paused=False,
            event_type="step_skipped",
            payload={"reason": request.reason, "actor": request.actor},
            step_id=step.step_id,
            clear_pending=True,
            clear_stop_reason=True,
        )

    def build_view(self, mission: AutonomousMission) -> MissionControlView:
        available_actions = ["pause", "cancel"]
        if mission.state.status in {MissionStatus.PAUSED, MissionStatus.STOPPED} and mission.state.pending_approval_step_id is None:
            available_actions.append("resume")
        if mission.state.pending_approval_step_id:
            available_actions.extend(["approve", "reject", "skip"])
        if mission.state.active_step_id:
            available_actions.append("retry-step")
        return MissionControlView(
            mission_id=mission.mission_id,
            status=mission.state.status,
            paused=mission.state.paused,
            waiting_for_confirmation=mission.state.waiting_for_confirmation,
            pending_approval_step_id=mission.state.pending_approval_step_id,
            active_step_id=mission.state.active_step_id,
            available_actions=available_actions,
            last_decision=mission.approval_history[0] if mission.approval_history else None,
            approval_history=mission.approval_history[:10],
            verification_summary=mission.verification_summary,
            recent_events=mission.control_events[-10:],
            metadata={"goal": mission.goal.objective},
        )

    def _mark_approved(self, mission: AutonomousMission, approval: MissionApprovalRecord) -> AutonomousMission:
        return self._update_state(
            mission,
            MissionStatus.RUNNING,
            paused=False,
            event_type="approved",
            payload={"actor": approval.actor, "reason": approval.reason, "decision": approval.decision.value},
            step_id=approval.step_id,
            clear_pending=True,
            clear_stop_reason=True,
        )

    def _mark_rejected(self, mission: AutonomousMission, approval: MissionApprovalRecord) -> AutonomousMission:
        step = self._resolve_step(mission, approval.step_id)
        step.status = MissionStepStatus.SKIPPED
        mission.step_results.append(
            MissionStepResult(
                mission_id=mission.mission_id,
                step_id=step.step_id,
                status=MissionStepStatus.SKIPPED,
                message=approval.reason or "step rejected by operator",
                data={"actor": approval.actor, "decision": approval.decision.value, "source": "mission_control"},
            )
        )
        return self._update_state(
            mission,
            MissionStatus.RUNNING,
            paused=False,
            event_type="rejected",
            payload={"actor": approval.actor, "reason": approval.reason, "decision": approval.decision.value},
            step_id=step.step_id,
            clear_pending=True,
            clear_stop_reason=True,
        )

    def _update_state(
        self,
        mission: AutonomousMission,
        status: MissionStatus,
        *,
        paused: bool,
        event_type: str,
        payload: dict[str, object],
        step_id: str | None,
        stop_reason: StopReason | None = None,
        last_error: str | None = None,
        clear_pending: bool = False,
        clear_stop_reason: bool = False,
    ) -> AutonomousMission:
        now = datetime.now(timezone.utc)
        state_update: dict[str, object] = {
            "status": status,
            "paused": paused,
            "waiting_for_confirmation": False if clear_pending else mission.state.waiting_for_confirmation,
            "updated_at": now,
            "last_error": last_error,
        }
        if clear_pending:
            state_update["pending_approval_step_id"] = None
            state_update["waiting_for_confirmation"] = False
        if clear_stop_reason:
            state_update["stop_reason"] = None
        elif stop_reason is not None:
            state_update["stop_reason"] = stop_reason
        mission.state = mission.state.model_copy(update=state_update)
        mission.updated_at = now
        self._state.save(mission)
        self._persistence.save_mission(mission)
        self._persistence.append_event(mission, event_type, step_id=step_id, payload=payload)
        return mission

    @staticmethod
    def _resolve_step(mission: AutonomousMission, step_id: str | None) -> MissionStep:
        if mission.plan is None or step_id is None:
            raise AutonomyValidationError("mission step not available", details={"mission_id": mission.mission_id, "step_id": step_id})
        for step in mission.plan.steps:
            if step.step_id == step_id:
                return step
        raise AutonomyValidationError("mission step not found", details={"mission_id": mission.mission_id, "step_id": step_id})
