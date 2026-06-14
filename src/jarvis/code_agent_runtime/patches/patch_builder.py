from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

from jarvis.code_agent_runtime.base import CodeActionKind, RiskLevel
from jarvis.code_agent_runtime.patches.diff_utils import sha256_text, unified_diff
from jarvis.code_agent_runtime.patches.models import PATCH_NOTICE, PatchFileChange, ProposedPatch
from jarvis.code_agent_runtime.patches.patch_store import PatchStore
from jarvis.code_agent_runtime.paths import normalize_project_path, relative_to_root
from jarvis.code_agent_runtime.security.path_policy import PathPolicy


class PatchBuilder:
    max_content_chars = 80_000
    max_diff_chars = 20_000

    def __init__(self, project_root: Path, store: PatchStore) -> None:
        self._root = project_root.resolve(strict=False)
        self._store = store
        self._path_policy = PathPolicy(self._root)

    def propose_replace(self, file: str, old_text: str, new_text: str, *, task: str = "replace text", skills: list[str] | None = None) -> dict[str, Any]:
        target = self._validate_target(file)
        before = self._read_existing(target)
        if old_text not in before:
            return self._blocked("old_text was not found in target file", [file])
        after = before.replace(old_text, new_text, 1)
        return self._propose(task, [self._change(target, before, after, "replace")], skills=skills or [])

    def propose_insert_after(self, file: str, anchor: str, text: str, *, task: str = "insert text after anchor", skills: list[str] | None = None) -> dict[str, Any]:
        target = self._validate_target(file)
        before = self._read_existing(target)
        if anchor not in before:
            return self._blocked("anchor was not found in target file", [file])
        after = before.replace(anchor, f"{anchor}{text}", 1)
        return self._propose(task, [self._change(target, before, after, "insert_after")], skills=skills or [])

    def propose_insert_before(self, file: str, anchor: str, text: str, *, task: str = "insert text before anchor", skills: list[str] | None = None) -> dict[str, Any]:
        target = self._validate_target(file)
        before = self._read_existing(target)
        if anchor not in before:
            return self._blocked("anchor was not found in target file", [file])
        after = before.replace(anchor, f"{text}{anchor}", 1)
        return self._propose(task, [self._change(target, before, after, "insert_before")], skills=skills or [])

    def propose_append(self, file: str, text: str, *, task: str = "append text", skills: list[str] | None = None) -> dict[str, Any]:
        target = self._validate_target(file)
        before = self._read_existing(target)
        after = before + text
        return self._propose(task, [self._change(target, before, after, "append")], skills=skills or [])

    def propose_create_file(self, file: str, content: str, *, task: str = "create file", skills: list[str] | None = None) -> dict[str, Any]:
        target = self._validate_target(file, may_create=True)
        if target.exists():
            return self._blocked("target file already exists; use replace/append operation", [file])
        return self._propose(task, [self._change(target, "", content, "create_file", existed=False)], skills=skills or [])

    def propose_unified_diff(self, diff_text: str, *, task: str = "unified diff", skills: list[str] | None = None) -> dict[str, Any]:
        return self._blocked("unified diff parsing is not enabled yet; use explicit patch operations", [])

    def _propose(self, task: str, changes: list[PatchFileChange], *, skills: list[str]) -> dict[str, Any]:
        if any(self._looks_sensitive(change.new_content) for change in changes):
            return self._blocked("patch content appears to contain secrets and was not stored", [change.path for change in changes])
        diff = "\n".join(change.unified_diff for change in changes)
        warnings: list[str] = []
        if len(diff) > self.max_diff_chars:
            warnings.append("large patch requires manual review")
        if any(len(change.new_content) > self.max_content_chars for change in changes):
            warnings.append("large file content requires manual review")
        risk = self._risk(changes, bool(warnings))
        patch_id = self._patch_id(task, changes)
        patch = ProposedPatch(
            id=patch_id,
            task=self._sanitize(task),
            target_files=[change.path for change in changes],
            summary=f"Proposed {', '.join(change.operation for change in changes)} for {len(changes)} file(s)",
            changes=changes,
            unified_diff=diff[: self.max_diff_chars],
            risk_level=int(risk),
            skills=skills,
            context_used=["explicit_patch_operation"],
            created_at=datetime.now(timezone.utc).isoformat(),
            requires_confirmation=risk >= RiskLevel.SENSITIVE,
            requires_pin=risk >= RiskLevel.CRITICAL,
            warnings=warnings,
        )
        saved = self._store.save(patch)
        return saved | {"status": patch.status, "notice": PATCH_NOTICE}

    def _change(self, target: Path, before: str, after: str, operation: str, *, existed: bool = True) -> PatchFileChange:
        self._validate_content(after)
        rel = relative_to_root(self._root, target)
        diff, truncated = unified_diff(rel, before, after, max_chars=self.max_diff_chars)
        if truncated:
            diff += "\n[warning] large patch requires manual review"
        return PatchFileChange(path=rel, operation=operation, original_hash=sha256_text(before), new_content=after, unified_diff=diff, existed=existed)

    def _validate_target(self, file: str, *, may_create: bool = False) -> Path:
        target = normalize_project_path(self._root, file)
        if not self._path_policy.is_allowed_project_path(target):
            raise PermissionError(f"path is outside project root: {file}")
        if self._path_policy.is_sensitive(target):
            raise PermissionError(f"sensitive file is blocked: {file}")
        if not may_create and not target.exists():
            raise FileNotFoundError(f"target file not found: {file}")
        return target

    def _read_existing(self, target: Path) -> str:
        data = target.read_bytes()
        if b"\x00" in data[:4096]:
            raise ValueError("binary files cannot be patched")
        return data.decode("utf-8", errors="replace")

    def _validate_content(self, content: str) -> None:
        if len(content) > self.max_content_chars * 2:
            raise ValueError("patch content is too large")
        if self._looks_sensitive(content):
            raise ValueError("patch content appears to contain secrets")

    @staticmethod
    def _looks_sensitive(value: str) -> bool:
        folded = value.casefold()
        return any(token in folded for token in (".env", "secret", "token", "credential", "password", "private key", "apikey", "api_key", "api key", "certificate", "id_rsa", ".pem", ".key"))

    @staticmethod
    def _sanitize(value: str) -> str:
        return "[redacted]" if PatchBuilder._looks_sensitive(value) else value[:1000]

    @staticmethod
    def _risk(changes: list[PatchFileChange], has_warnings: bool) -> RiskLevel:
        if has_warnings:
            return RiskLevel.SENSITIVE
        if any(change.existed for change in changes):
            return RiskLevel.SENSITIVE
        return RiskLevel.MINOR_CHANGE

    @staticmethod
    def _patch_id(task: str, changes: list[PatchFileChange]) -> str:
        raw = f"{datetime.now(timezone.utc).isoformat()}:{task}:{','.join(change.path for change in changes)}"
        return f"patch-{sha1(raw.encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def _blocked(message: str, targets: list[str]) -> dict[str, Any]:
        return {"status": "blocked", "message": message, "target_files": targets, "notice": PATCH_NOTICE}
