from __future__ import annotations

from datetime import datetime, timezone

from .base import AutonomousMission, MissionState, MissionStatus, StopReason


class MissionStateManager:
    def __init__(self) -> None:
        self._missions: dict[str, AutonomousMission] = {}
        self._active_mission_id: str | None = None

    def save(self, mission: AutonomousMission) -> AutonomousMission:
        mission.updated_at = datetime.now(timezone.utc)
        self._missions[mission.mission_id] = mission
        if mission.state.status in {
            MissionStatus.RUNNING,
            MissionStatus.PLANNING,
            MissionStatus.WAITING_CONFIRMATION,
            MissionStatus.AWAITING_REVIEW,
            MissionStatus.PAUSED,
        }:
            self._active_mission_id = mission.mission_id
        elif self._active_mission_id == mission.mission_id and mission.state.status in {
            MissionStatus.COMPLETED,
            MissionStatus.FAILED,
            MissionStatus.CANCELLED,
            MissionStatus.STOPPED,
        }:
            self._active_mission_id = None
        return mission

    def hydrate(self, mission: AutonomousMission) -> AutonomousMission:
        return self.save(mission)

    def get(self, mission_id: str) -> AutonomousMission | None:
        return self._missions.get(mission_id)

    def list(self) -> list[AutonomousMission]:
        return sorted(self._missions.values(), key=lambda item: item.created_at, reverse=True)

    def active(self) -> AutonomousMission | None:
        if self._active_mission_id is None:
            return None
        return self._missions.get(self._active_mission_id)

    def cancel(self, mission_id: str) -> AutonomousMission | None:
        mission = self._missions.get(mission_id)
        if mission is None:
            return None
        mission.state = mission.state.model_copy(update={"status": MissionStatus.CANCELLED, "stop_reason": StopReason.CANCELLED, "updated_at": datetime.now(timezone.utc)})
        self.save(mission)
        return mission
