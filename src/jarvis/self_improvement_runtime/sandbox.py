from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .models import SelfImprovementProposal, SelfImprovementSandboxRecord


class SelfImprovementSandbox:
    def __init__(self, *, data_dir: Path, logger: logging.Logger | None = None) -> None:
        self._root = (data_dir / "self_improvement").resolve()
        self._sandboxes = self._root / "sandboxes"
        self._history = self._root / "history"
        self._backups = self._root / "backups"
        self._logger = logger or logging.getLogger("jarvis.self_improvement.sandbox")
        self._sandboxes.mkdir(parents=True, exist_ok=True)
        self._history.mkdir(parents=True, exist_ok=True)
        self._backups.mkdir(parents=True, exist_ok=True)

    def create(self, *, workspace_root: Path, session_id: str) -> Path:
        sandbox_root = self._sandboxes / session_id
        if sandbox_root.exists():
            shutil.rmtree(sandbox_root)
        ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "runtime", ".git")
        shutil.copytree(workspace_root, sandbox_root, ignore=ignore)
        return sandbox_root

    def apply_patch_sandbox(self, *, sandbox_root: Path, workspace_root: Path, proposal: SelfImprovementProposal) -> SelfImprovementSandboxRecord:
        source_root = workspace_root.resolve()
        source_file = Path(proposal.file_path).resolve()
        if source_root not in source_file.parents and source_file != source_root:
            raise ValueError("proposal points outside workspace root")
        relative = source_file.relative_to(source_root)
        sandbox_file = (sandbox_root / relative).resolve()
        sandbox_file.parent.mkdir(parents=True, exist_ok=True)
        sandbox_file.write_text(proposal.updated_text, encoding="utf-8")
        return SelfImprovementSandboxRecord(sandbox_root=str(sandbox_root), changed_files=(str(relative).replace("\\", "/"),))

    def persist_backup(self, *, session_id: str, workspace_root: Path, proposal: SelfImprovementProposal) -> dict[str, object]:
        source_root = workspace_root.resolve()
        source_file = Path(proposal.file_path).resolve()
        relative = source_file.relative_to(source_root)
        backup_dir = self._backups / session_id
        backup_file = backup_dir / relative
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        existed = source_file.exists()
        if existed:
            backup_file.write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = {
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(source_root),
            "proposal": {
                "file_path": str(relative).replace("\\", "/"),
                "summary": proposal.summary,
            },
            "backups": [{"relative_path": str(relative).replace("\\", "/"), "backup_path": str(backup_file), "existed": existed}],
            "applied": False,
        }
        (self._history / f"{session_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def backup_ready(self, session_id: str) -> bool:
        manifest_path = self._history / f"{session_id}.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return bool(manifest.get("backups"))

    def apply_to_workspace(self, *, session_id: str, workspace_root: Path, proposal: SelfImprovementProposal) -> dict[str, object]:
        source_root = workspace_root.resolve()
        source_file = Path(proposal.file_path).resolve()
        relative = source_file.relative_to(source_root)
        target = (source_root / relative).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f"{target.name}.jarvis-tmp")
        try:
            temp_target.write_text(proposal.updated_text, encoding="utf-8")
            temp_target.replace(target)
            manifest_path = self._history / f"{session_id}.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["applied"] = True
                manifest["applied_at"] = datetime.now(timezone.utc).isoformat()
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return {"applied": 1, "session_id": session_id, "target": str(target)}
        except Exception:
            if temp_target.exists():
                temp_target.unlink()
            if self.backup_ready(session_id):
                self.rollback(session_id)
            raise

    def rollback(self, session_id: str) -> dict[str, object]:
        manifest_path = self._history / f"{session_id}.json"
        if not manifest_path.exists():
            return {"restored": 0, "session_id": session_id}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored = 0
        workspace_root = Path(manifest["workspace_root"]).resolve()
        for item in manifest.get("backups", []):
            relative_path = Path(item["relative_path"])
            target = (workspace_root / relative_path).resolve()
            backup_path = Path(item["backup_path"]).resolve()
            if item.get("existed", False) and backup_path.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
                restored += 1
            elif target.exists():
                target.unlink()
                restored += 1
        manifest["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
        manifest["applied"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"restored": restored, "session_id": session_id}
