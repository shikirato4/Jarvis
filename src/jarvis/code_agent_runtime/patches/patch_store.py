from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any

from jarvis.code_agent_runtime.patches.models import ProposedPatch


class PatchStore:
    def __init__(self, root: Path, *, max_patch_bytes: int = 500_000) -> None:
        self.root = root
        self.max_patch_bytes = max_patch_bytes

    def save(self, patch: ProposedPatch) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(patch.id)
        payload = patch.to_dict()
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if len(text.encode("utf-8")) > self.max_patch_bytes:
            patch.warnings.append("patch exceeded storage size limit; diff/content should be reduced")
            payload = patch.to_dict()
            text = json.dumps(payload, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        return patch.to_dict(include_content=False) | {"path": str(path)}

    def load(self, patch_id: str) -> ProposedPatch:
        path = self._path(patch_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            self._backup_corrupt(path)
            raise ValueError(f"patch is missing or corrupt: {patch_id}") from exc
        return ProposedPatch.from_dict(payload)

    def show(self, patch_id: str) -> dict[str, Any]:
        return self.load(patch_id).to_dict(include_content=False)

    def list(self, *, limit: int = 100) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        patches = []
        warnings = []
        for path in sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            if len(patches) >= limit:
                break
            try:
                patch = ProposedPatch.from_dict(json.loads(path.read_text(encoding="utf-8")))
                patches.append(patch.to_dict(include_content=False))
            except (json.JSONDecodeError, OSError, TypeError):
                backup = self._backup_corrupt(path)
                warnings.append({"message": "patch file was corrupt and was backed up", "backup_path": str(backup)})
        return {"patch_count": len(patches), "patches": patches, "warnings": warnings}

    def update_status(self, patch_id: str, status: str, *, warning: str = "") -> dict[str, Any]:
        patch = self.load(patch_id)
        patch.status = status
        if warning:
            patch.warnings.append(warning)
        return self.save(patch)

    def stats(self) -> dict[str, Any]:
        listing = self.list(limit=10_000)
        by_status: dict[str, int] = {}
        for patch in listing["patches"]:
            status = str(patch.get("status", "unknown"))
            by_status[status] = by_status.get(status, 0) + 1
        return {"patch_count": listing["patch_count"], "by_status": dict(sorted(by_status.items())), "warnings": listing["warnings"], "path": str(self.root)}

    def _path(self, patch_id: str) -> Path:
        safe = "".join(char for char in patch_id if char.isalnum() or char in "-_")
        if not safe:
            raise ValueError("invalid patch id")
        return self.root / f"{safe}.json"

    @staticmethod
    def _stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _backup_corrupt(self, path: Path) -> Path:
        backup = path.with_name(f"{path.name}.corrupt-{self._stamp()}.bak")
        if path.exists():
            try:
                copy2(path, backup)
            except OSError:
                pass
        return backup
