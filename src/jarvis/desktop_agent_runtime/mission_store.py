from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

from .models import DesktopAgentMissionReceipt


class DesktopAgentMissionStore:
    def __init__(self, data_dir: Path) -> None:
        self._root = Path(data_dir) / "desktop_agent_missions"
        self._lock = RLock()

    def ensure_ready(self) -> None:
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, mission: DesktopAgentMissionReceipt) -> DesktopAgentMissionReceipt:
        payload = mission.model_dump(mode="json")
        with self._lock:
            self.ensure_ready()
            path = self._root / f"{mission.mission_id}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        return mission

    def load(self, mission_id: str) -> DesktopAgentMissionReceipt | None:
        path = self._root / f"{mission_id}.json"
        if not path.exists():
            return None
        with self._lock:
            return DesktopAgentMissionReceipt.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[DesktopAgentMissionReceipt]:
        if not self._root.exists():
            return []
        with self._lock:
            items: list[DesktopAgentMissionReceipt] = []
            for path in sorted(self._root.glob("*.json"), key=lambda item: item.stat().st_mtime):
                items.append(DesktopAgentMissionReceipt.model_validate_json(path.read_text(encoding="utf-8")))
            return items
