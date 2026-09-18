from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class WindowsErrorCode(StrEnum):
    TARGET_NOT_FOUND = "target_not_found"
    AMBIGUOUS_TARGET = "ambiguous_target"
    TARGET_STALE = "target_stale"
    TARGET_CLOSED = "target_closed"
    PERMISSION_DENIED = "permission_denied"
    OPERATION_NOT_SUPPORTED = "operation_not_supported"
    LAUNCH_FAILED = "launch_failed"
    FOCUS_FAILED = "focus_failed"
    WINDOWS_API_ERROR = "windows_api_error"
    TIMEOUT = "timeout"


class WindowsRuntimeError(RuntimeError):
    def __init__(
        self,
        code: WindowsErrorCode,
        message: str,
        *,
        candidates: list[WindowRef] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.candidates = candidates or []


class WindowBounds(BaseModel):
    left: int
    top: int
    right: int
    bottom: int


class WindowRef(BaseModel):
    runtime_id: str = Field(description="Opaque runtime window identity; prefer this for follow-up actions.")
    handle: str = Field(description="Native HWND identity captured with the process id.")
    pid: int
    process_name: str | None = None
    application: str | None = None
    title: str
    class_name: str | None = None
    visible: bool = True
    minimized: bool = False
    maximized: bool = False
    foreground: bool = False
    bounds: WindowBounds | None = None


class WindowTarget(BaseModel):
    runtime_id: str | None = Field(
        default=None,
        description="Exact runtime_id returned by a recent Windows window observation.",
    )
    pid: int | None = Field(default=None, ge=1)
    process_name: str | None = None
    application: str | None = None
    title: str | None = None
    class_name: str | None = None
    query: str | None = Field(
        default=None,
        description="Semantic app/window name used only when an exact structured identity is unavailable.",
    )

    @model_validator(mode="after")
    def require_selector(self) -> WindowTarget:
        if not any(
            (
                self.runtime_id,
                self.pid,
                self.process_name,
                self.application,
                self.title,
                self.class_name,
                self.query,
            )
        ):
            raise ValueError("at least one window selector is required")
        return self


class WindowsEnvironment(BaseModel):
    edition: str
    version: str
    build: str
    architecture: str
    platform: str = "Windows"
    interactive_session: bool
    session_id: int | None = None


class ProcessRef(BaseModel):
    pid: int
    process_name: str
    application: str | None = None
    session_id: int | None = None
    interactive: bool = False
    visible_window_count: int = 0
    executable_name: str | None = None


class WindowsObservation(BaseModel):
    status: str = "success"
    operation: str
    verified: bool
    target: WindowRef | None = None
    before: WindowRef | None = None
    after: WindowRef | None = None
    environment: WindowsEnvironment | None = None
    processes: list[ProcessRef] = Field(default_factory=list)
    windows: list[WindowRef] = Field(default_factory=list)
    count: int | None = None
    truncated: bool = False
    error_code: WindowsErrorCode | None = None
    message: str | None = None
    candidates: list[WindowRef] = Field(default_factory=list)
