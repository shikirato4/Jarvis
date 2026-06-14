from __future__ import annotations

import json
from pathlib import Path

from jarvis.code_agent_runtime.base import CodeAgentReceipt

REDACTED = "[redacted]"


class ActionLog:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    @property
    def path(self) -> Path:
        return self._log_path

    def append(self, receipt: CodeAgentReceipt) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "started_at": receipt.started_at.isoformat(),
            "finished_at": receipt.finished_at.isoformat(),
            "mode": receipt.mode.value,
            "action": receipt.action.value,
            "tool": receipt.tool,
            "risk_level": int(receipt.risk.level),
            "risk_reason": receipt.risk.reason,
            "target": self._redact(receipt.target),
            "commands": [self._redact(command) for command in receipt.commands],
            "touched_files": [self._redact(path) for path in receipt.touched_files],
            "status": receipt.status.value,
            "result": receipt.message,
            "blocked_reason": receipt.blocked_reason,
            "confirmation_required": receipt.confirmation_required,
            "pin_required": receipt.pin_required,
            "pin_verified": receipt.pin_verified,
            "errors": [self._redact(error) for error in receipt.errors],
        }
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    @staticmethod
    def _redact(value: str | None) -> str | None:
        if value is None:
            return None
        folded = value.casefold()
        if any(token in folded for token in (".env", "secret", "token", "credential", "password", "private", "apikey", "api_key")):
            return REDACTED
        return value
