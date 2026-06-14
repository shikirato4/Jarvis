from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class RiskLevel(IntEnum):
    SAFE = 0
    MINOR_CHANGE = 1
    SENSITIVE = 2
    CRITICAL = 3


class OperationMode(StrEnum):
    CONVERSATION = "conversation"
    PROGRAMMER = "programmer"
    ADMIN = "admin"


class CodeActionKind(StrEnum):
    PROJECT_SCAN = "project_scan"
    FILE_READ = "file_read"
    PROJECT_SEARCH = "project_search"
    FILE_WRITE = "file_write"
    COMMAND_RUN = "command_run"
    GIT_OPERATION = "git_operation"
    PIN_CONFIGURE = "pin_configure"
    PIN_CHANGE = "pin_change"
    PIN_VERIFY = "pin_verify"
    MODE_READ = "mode_read"
    MODE_CHANGE = "mode_change"


class CodeActionStatus(StrEnum):
    OK = "ok"
    BLOCKED = "blocked"
    CONFIRMATION_REQUIRED = "confirmation_required"
    FAILED = "failed"


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"
    REQUIRE_PIN = "require_pin"


class RiskAssessment(JarvisBaseModel):
    level: RiskLevel
    reason: str
    requires_confirmation: bool = False
    requires_pin: bool = False
    tags: list[str] = Field(default_factory=list)


class PermissionResult(JarvisBaseModel):
    decision: PermissionDecision
    allowed: bool = False
    reason: str
    requires_confirmation: bool = False
    requires_pin: bool = False
    pin_verified: bool | None = None
    mode: OperationMode
    confirmation_prompt: dict[str, Any] | None = None


class AuthorizationContext(JarvisBaseModel):
    action: CodeActionKind
    target: str
    risk: RiskAssessment
    mode: OperationMode
    allowed: bool
    confirmation_confirmed: bool = False
    pin_verified: bool | None = None


class CodeAgentReceipt(JarvisBaseModel):
    action: CodeActionKind
    status: CodeActionStatus
    risk: RiskAssessment
    mode: OperationMode = OperationMode.PROGRAMMER
    message: str
    target: str | None = None
    touched_files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    confirmation_required: bool = False
    pin_required: bool = False
    pin_verified: bool | None = None
    blocked_reason: str | None = None
    tool: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProjectFileSummary(JarvisBaseModel):
    path: str
    size_bytes: int
    extension: str


class ProjectScanResult(JarvisBaseModel):
    root: str
    directories: list[str] = Field(default_factory=list)
    files: list[ProjectFileSummary] = Field(default_factory=list)
    ignored_directories: list[str] = Field(default_factory=list)
    total_files_seen: int = 0
    total_files_listed: int = 0
    extension_counts: dict[str, int] = Field(default_factory=dict)
    summary: str


class FileReadResult(JarvisBaseModel):
    path: str
    size_bytes: int
    truncated: bool = False
    content: str


class SearchMatch(JarvisBaseModel):
    path: str
    line_number: int | None = None
    snippet: str
    match_type: str


class SearchResult(JarvisBaseModel):
    root: str
    query: str
    mode: str
    matches: list[SearchMatch] = Field(default_factory=list)
    scanned_files: int = 0
    skipped_files: int = 0


class CommandRunResult(JarvisBaseModel):
    command: str
    cwd: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CodeTaskRequest(JarvisBaseModel):
    task: str
    path: str | None = None
    content: str | None = None
    command: str | None = None
    query: str | None = None
    confirm: bool = False
    pin: str | None = None
    dry_run: bool = False


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
