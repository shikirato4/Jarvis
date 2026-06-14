from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PATCH_NOTICE = "Review this diff before applying. Patches are explicit changes only; do not copy external code blindly."


@dataclass(frozen=True)
class PatchFileChange:
    path: str
    operation: str
    original_hash: str
    new_content: str
    unified_diff: str
    existed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation,
            "original_hash": self.original_hash,
            "new_content": self.new_content,
            "unified_diff": self.unified_diff,
            "existed": self.existed,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "PatchFileChange":
        return PatchFileChange(
            path=str(payload.get("path", "")),
            operation=str(payload.get("operation", "")),
            original_hash=str(payload.get("original_hash", "")),
            new_content=str(payload.get("new_content", "")),
            unified_diff=str(payload.get("unified_diff", "")),
            existed=bool(payload.get("existed", True)),
        )


@dataclass(frozen=True)
class PatchHunk:
    file: str
    header: str
    body: str

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "header": self.header, "body": self.body}


@dataclass
class ProposedPatch:
    id: str
    task: str
    target_files: list[str]
    summary: str
    changes: list[PatchFileChange]
    unified_diff: str
    risk_level: int
    skills: list[str] = field(default_factory=list)
    context_used: list[str] = field(default_factory=list)
    created_at: str = ""
    status: str = "proposed"
    requires_confirmation: bool = False
    requires_pin: bool = False
    warnings: list[str] = field(default_factory=list)
    notice: str = PATCH_NOTICE

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        changes = [change.to_dict() for change in self.changes]
        if not include_content:
            changes = [{key: value for key, value in change.items() if key != "new_content"} for change in changes]
        return {
            "id": self.id,
            "task": self.task,
            "target_files": self.target_files,
            "summary": self.summary,
            "changes": changes,
            "unified_diff": self.unified_diff,
            "risk_level": self.risk_level,
            "skills": self.skills,
            "context_used": self.context_used,
            "created_at": self.created_at,
            "status": self.status,
            "requires_confirmation": self.requires_confirmation,
            "requires_pin": self.requires_pin,
            "warnings": self.warnings,
            "notice": self.notice,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ProposedPatch":
        return ProposedPatch(
            id=str(payload.get("id", "")),
            task=str(payload.get("task", "")),
            target_files=[str(item) for item in payload.get("target_files", [])],
            summary=str(payload.get("summary", "")),
            changes=[PatchFileChange.from_dict(item) for item in payload.get("changes", []) if isinstance(item, dict)],
            unified_diff=str(payload.get("unified_diff", "")),
            risk_level=int(payload.get("risk_level", 0) or 0),
            skills=[str(item) for item in payload.get("skills", [])],
            context_used=[str(item) for item in payload.get("context_used", [])],
            created_at=str(payload.get("created_at", "")),
            status=str(payload.get("status", "proposed")),
            requires_confirmation=bool(payload.get("requires_confirmation", False)),
            requires_pin=bool(payload.get("requires_pin", False)),
            warnings=[str(item) for item in payload.get("warnings", [])],
        )


@dataclass(frozen=True)
class PatchReview:
    patch_id: str
    status: str
    warnings: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"patch_id": self.patch_id, "status": self.status, "warnings": self.warnings, "summary": self.summary}


@dataclass(frozen=True)
class PatchApplyResult:
    patch_id: str
    status: str
    touched_files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    message: str = ""
    git_diff_stat: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "status": self.status,
            "touched_files": self.touched_files,
            "commands": self.commands,
            "message": self.message,
            "git_diff_stat": self.git_diff_stat,
            "errors": self.errors,
            "notice": PATCH_NOTICE,
        }
