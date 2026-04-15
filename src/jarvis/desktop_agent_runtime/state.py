from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from .models import DesktopAgentMissionReceipt


@dataclass
class DesktopAgentStateStore:
    _lock: RLock = field(default_factory=RLock)
    _missions: dict[str, DesktopAgentMissionReceipt] = field(default_factory=dict)

    def save(self, receipt: DesktopAgentMissionReceipt) -> DesktopAgentMissionReceipt:
        with self._lock:
            self._missions[receipt.mission_id] = receipt
        return receipt

    def seed(self, receipts: list[DesktopAgentMissionReceipt]) -> None:
        with self._lock:
            self._missions = {receipt.mission_id: receipt for receipt in receipts}

    def get(self, mission_id: str) -> DesktopAgentMissionReceipt | None:
        with self._lock:
            return self._missions.get(mission_id)

    def latest(self) -> DesktopAgentMissionReceipt | None:
        with self._lock:
            if not self._missions:
                return None
            return list(self._missions.values())[-1]

    def list(self) -> list[DesktopAgentMissionReceipt]:
        with self._lock:
            return list(self._missions.values())
