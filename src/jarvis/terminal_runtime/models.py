from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ShellKind(StrEnum):
    POWERSHELL = "powershell"
    CMD = "cmd"


class TerminalSessionState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    STOPPING = "stopping"
    CLOSED = "closed"
    FAILED = "failed"


class ExecutionState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class TerminalErrorCode(StrEnum):
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    INVALID_CWD = "INVALID_CWD"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"
    PROCESS_FAILED = "PROCESS_FAILED"
    SESSION_CLOSED = "SESSION_CLOSED"
    SESSION_BUSY = "SESSION_BUSY"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SHELL_UNAVAILABLE = "SHELL_UNAVAILABLE"
    SPAWN_FAILED = "SPAWN_FAILED"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"
    INPUT_NOT_ACCEPTED = "INPUT_NOT_ACCEPTED"
    WINDOWS_TERMINAL_UNAVAILABLE = "WINDOWS_TERMINAL_UNAVAILABLE"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"


class TerminalError(RuntimeError):
    def __init__(self, code: TerminalErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class TerminalOutput(JarvisBaseModel):
    sequence: int
    stream: str
    text: str
    timestamp: datetime = Field(default_factory=utc_now)


class TerminalResult(JarvisBaseModel):
    session_id: str
    execution_id: str
    shell: ShellKind
    cwd: str
    status: ExecutionState
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output: list[TerminalOutput] = Field(default_factory=list)
    duration_ms: float = 0.0
    timeout_seconds: float | None = None
    truncated: bool = False
    error_code: TerminalErrorCode | None = None


class TerminalSessionSnapshot(JarvisBaseModel):
    session_id: str
    shell: ShellKind
    cwd: str
    state: TerminalSessionState
    owned_by_jarvis: bool
    pid: int | None
    created_at: datetime
    last_activity: datetime
    active_execution_id: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)


class VisibleTerminalResult(JarvisBaseModel):
    pid: int
    shell: ShellKind
    cwd: str
    owned_by_jarvis: bool = True
    launched: bool = True
    verification: str = "launcher accepted request"
