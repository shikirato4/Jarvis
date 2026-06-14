from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

from jarvis.code_agent_runtime.repo_library.models import RepoRecord


class RepoLibraryStorage:
    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, Any]:
        if not self._path.exists():
            return self._default()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            return self._recover_corrupt()
        payload.setdefault("schema_version", 1)
        payload.setdefault("library_root", "")
        payload.setdefault("indexed_at", "")
        payload.setdefault("repos", [])
        payload["repos"] = [item for item in payload["repos"] if isinstance(item, dict)]
        return payload

    def save(self, *, library_root: Path, repos: list[RepoRecord]) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "library_root": str(library_root.resolve(strict=False)),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "repo_count": len(repos),
            "repos": [repo.to_dict() for repo in repos],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def records(self) -> list[RepoRecord]:
        return [RepoRecord.from_dict(item) for item in self.load().get("repos", [])]

    @staticmethod
    def _default() -> dict[str, Any]:
        return {"schema_version": 1, "library_root": "", "indexed_at": "", "repo_count": 0, "repos": [], "warnings": []}

    def _recover_corrupt(self) -> dict[str, Any]:
        payload = self._default()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = self._path.with_name(f"{self._path.name}.corrupt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak")
        if self._path.exists():
            try:
                copy2(self._path, backup_path)
                payload["warnings"].append({"message": "repo library index was corrupt and was reset", "backup_path": str(backup_path)})
            except OSError as exc:
                payload["warnings"].append({"message": f"repo library index was corrupt and backup failed: {exc}"})
        return payload
