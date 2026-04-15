from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import (
    ApprovalDecision,
    AutonomousMission,
    MissionApprovalRecord,
    MissionPersistenceSnapshot,
    MissionStatus,
)
from .repository import MissionRepository


class MissionPersistenceService:
    def __init__(self, repository: MissionRepository, *, logger=None) -> None:
        self._repository = repository
        self._logger = logger

    def create_schema(self) -> None:
        self._repository.create_schema()

    def save_mission(self, mission: AutonomousMission) -> AutonomousMission:
        payload = mission.model_dump(mode="json")
        self._repository.upsert_mission(
            mission_id=mission.mission_id,
            status=mission.state.status.value,
            goal=mission.goal.objective,
            active_step_id=mission.state.active_step_id,
            requires_human_attention=mission.state.waiting_for_confirmation or mission.state.pending_approval_step_id is not None,
            snapshot=payload,
            created_at=mission.created_at,
            updated_at=mission.updated_at,
        )
        return mission

    def append_event(
        self,
        mission: AutonomousMission,
        event_type: str,
        *,
        step_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        record = self._repository.append_event(
            mission_id=mission.mission_id,
            event_type=event_type,
            step_id=step_id,
            payload=payload,
        )
        event = {
            "event_type": record.event_type,
            "step_id": record.step_id,
            "payload": payload,
            "created_at": record.created_at.isoformat(),
        }
        mission.control_events.append(event)
        mission.updated_at = datetime.now(timezone.utc)
        self.save_mission(mission)
        return event

    def record_approval(self, mission: AutonomousMission, approval: MissionApprovalRecord) -> MissionApprovalRecord:
        record = self._repository.record_approval(
            mission_id=approval.mission_id,
            step_id=approval.step_id,
            decision=approval.decision.value,
            actor=approval.actor,
            reason=approval.reason,
            metadata=approval.metadata,
        )
        stored = MissionApprovalRecord(
            mission_id=record.mission_id,
            step_id=record.step_id,
            decision=ApprovalDecision(record.decision),
            actor=record.actor,
            reason=record.reason,
            created_at=record.created_at,
            metadata=self._repository.load_approval_metadata(record),
        )
        mission.approval_history.insert(0, stored)
        mission.updated_at = datetime.now(timezone.utc)
        self.save_mission(mission)
        return stored

    def load_mission(self, mission_id: str) -> AutonomousMission | None:
        snapshot = self._repository.load_snapshot(self._repository.get_mission(mission_id))
        if snapshot is None:
            return None
        return AutonomousMission.model_validate(snapshot)

    def list_missions(self) -> list[AutonomousMission]:
        missions: list[AutonomousMission] = []
        for record in self._repository.list_missions():
            snapshot = self._repository.load_snapshot(record)
            if snapshot is None:
                continue
            missions.append(AutonomousMission.model_validate(snapshot))
        return missions

    def load_snapshot(self, mission_id: str) -> MissionPersistenceSnapshot | None:
        mission = self.load_mission(mission_id)
        if mission is None:
            return None
        events = [
            {
                "event_type": record.event_type,
                "step_id": record.step_id,
                "payload": self._repository.load_event_payload(record),
                "created_at": record.created_at.isoformat(),
            }
            for record in self._repository.list_events(mission_id)
        ]
        approvals = [
            MissionApprovalRecord(
                mission_id=record.mission_id,
                step_id=record.step_id,
                decision=ApprovalDecision(record.decision),
                actor=record.actor,
                reason=record.reason,
                created_at=record.created_at,
                metadata=self._repository.load_approval_metadata(record),
            )
            for record in self._repository.list_approvals(mission_id)
        ]
        return MissionPersistenceSnapshot(mission=mission, events=events, approvals=approvals)

    def hydrate_state(self, state_manager) -> list[AutonomousMission]:
        hydrated: list[AutonomousMission] = []
        for record in self._repository.list_active_missions():
            snapshot = self._repository.load_snapshot(record)
            if snapshot is None:
                continue
            mission = AutonomousMission.model_validate(snapshot)
            mission = self._normalize_rehydrated_mission(mission)
            state_manager.hydrate(mission)
            self.save_mission(mission)
            hydrated.append(mission)
        return hydrated

    def _normalize_rehydrated_mission(self, mission: AutonomousMission) -> AutonomousMission:
        if mission.state.status == MissionStatus.RUNNING:
            mission.state = mission.state.model_copy(
                update={
                    "status": MissionStatus.PAUSED,
                    "paused": True,
                    "last_error": "mission rehydrated after restart; manual resume required",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self.append_event(
                mission,
                "paused",
                step_id=mission.state.active_step_id,
                payload={"reason": "rehydrated_running_mission", "source": "mission_persistence"},
            )
        elif mission.state.status == MissionStatus.PLANNING:
            mission.state = mission.state.model_copy(
                update={
                    "status": MissionStatus.PAUSED,
                    "paused": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        return mission
