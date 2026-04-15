from __future__ import annotations

from jarvis.autonomy.base import MissionApprovalRequest, MissionControlActionRequest

from .views import HudActionReceipt, HudMissionView


class HudMissionController:
    def __init__(self, autonomy_service) -> None:
        self._autonomy = autonomy_service

    def missions(self) -> list[HudMissionView]:
        views: list[HudMissionView] = []
        for mission in self._autonomy.list_missions():
            mission_id = str(mission["mission_id"])
            receipt = self._autonomy.inspect_mission(mission_id)
            control = self._autonomy.mission_control_view(mission_id)
            views.append(
                HudMissionView(
                    mission_id=mission_id,
                    goal=mission["goal"],
                    status=mission["status"],
                    autonomy_level=mission.get("autonomy_level"),
                    active_step_id=control.active_step_id,
                    pending_approval_step_id=control.pending_approval_step_id,
                    waiting_for_confirmation=control.waiting_for_confirmation,
                    paused=control.paused,
                    available_actions=control.available_actions,
                    current_step=receipt.current_step.model_dump(mode="json") if receipt.current_step else None,
                    verification_summary=control.verification_summary.model_dump(mode="json") if control.verification_summary else None,
                    recent_results=[item.model_dump(mode="json") for item in receipt.recent_results],
                    recent_events=list(control.recent_events),
                )
            )
        return views

    def approve(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None, actor: str = "hud") -> HudActionReceipt:
        receipt = self._autonomy.approve_step(MissionApprovalRequest(mission_id=mission_id, step_id=step_id, decision="approve", reason=reason, actor=actor))
        return HudActionReceipt(action_name="approve", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))

    def reject(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None, actor: str = "hud") -> HudActionReceipt:
        receipt = self._autonomy.reject_step(MissionApprovalRequest(mission_id=mission_id, step_id=step_id, decision="reject", reason=reason, actor=actor))
        return HudActionReceipt(action_name="reject", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))

    def pause(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None, actor: str = "hud") -> HudActionReceipt:
        receipt = self._autonomy.pause_mission(MissionControlActionRequest(mission_id=mission_id, step_id=step_id, reason=reason, actor=actor))
        return HudActionReceipt(action_name="pause", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))

    def resume(self, mission_id: str, *, step_id: str | None = None, reason: str | None = None, actor: str = "hud") -> HudActionReceipt:
        receipt = self._autonomy.resume_mission(MissionControlActionRequest(mission_id=mission_id, step_id=step_id, reason=reason, actor=actor))
        return HudActionReceipt(action_name="resume", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))

    def stop(self, mission_id: str) -> HudActionReceipt:
        receipt = self._autonomy.stop_mission(mission_id)
        return HudActionReceipt(action_name="stop", success=True, message=receipt.message, data=receipt.model_dump(mode="json"))
