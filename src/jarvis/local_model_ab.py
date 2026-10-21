"""Small, resumable result store for local-model A/B probes.

The store intentionally persists metadata only: prompts, histories, and model
reasoning are never written to the runtime evidence files.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Iterable


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if str(key).lower() not in {"reasoning_content", "raw_history", "messages"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ResumableABStore:
    """Append-only case evidence with immediate persistence and resume support."""

    def __init__(self, run_dir: Path, manifest: dict[str, Any] | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.results_path = self.run_dir / "results.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if manifest is not None and not self.manifest_path.exists():
            self.manifest_path.write_text(
                json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._refresh_summary()

    def _records(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self._records():
            case_id = str(record.get("case_id", ""))
            if case_id:
                latest[case_id] = record
        return latest

    def completed_case_ids(self) -> set[str]:
        return {
            case_id
            for case_id, record in self.latest().items()
            if record.get("status") == "completed"
        }

    def append(self, result: dict[str, Any]) -> None:
        safe = _json_safe(dict(result))
        safe.setdefault("recorded_at", datetime.now(UTC).isoformat())
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        latest = self.latest()
        counts: dict[str, int] = {}
        for record in latest.values():
            status = str(record.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        summary = {
            "updated_at": datetime.now(UTC).isoformat(),
            "case_count": len(latest),
            "status_counts": counts,
            "completed_case_ids": sorted(
                case_id for case_id, record in latest.items() if record.get("status") == "completed"
            ),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def pending(self, cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        done = self.completed_case_ids()
        return [case for case in cases if str(case.get("case_id", "")) not in done]
