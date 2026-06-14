from __future__ import annotations

import json
from pathlib import Path

from jarvis.code_agent_runtime.base import OperationMode


class OperationModeStore:
    def __init__(self, path: Path, *, default_mode: OperationMode = OperationMode.PROGRAMMER) -> None:
        self._path = path
        self._default_mode = default_mode

    def current(self) -> OperationMode:
        if not self._path.exists():
            return self._default_mode
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return OperationMode(payload.get("mode", self._default_mode.value))
        except Exception:  # noqa: BLE001
            return self._default_mode

    def set(self, mode: OperationMode | str) -> OperationMode:
        selected = OperationMode(mode)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"mode": selected.value}, indent=2), encoding="utf-8")
        return selected
