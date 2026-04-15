from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, Field

from jarvis.models.base import JarvisBaseModel


class UIAutomationMode(StrEnum):
    COPILOT = "copilot"
    DIRECT = "direct"


class UIRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UIOperationStatus(StrEnum):
    EXECUTED = "executed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    BLOCKED = "blocked"


class MouseButton(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class WindowInfo(JarvisBaseModel):
    handle: str
    title: str
    class_name: str | None = None
    process_id: int | None = None
    process_name: str | None = None
    rect: dict[str, int] = Field(default_factory=dict)


class PointerPosition(JarvisBaseModel):
    x: int
    y: int


class WriteTextRequest(JarvisBaseModel):
    text: str
    mode: UIAutomationMode = UIAutomationMode.COPILOT
    block_size: int | None = None
    typing_interval_ms: int | None = None
    pause_between_blocks_ms: int | None = None
    focus_target: str | None = Field(default=None, validation_alias=AliasChoices("focus_target", "target_window"))
    ensure_window_contains: str | None = None
    approved: bool = False
    timeout_ms: int | None = None
    watchdog_timeout_ms: int | None = None


class InsertBlocksRequest(JarvisBaseModel):
    blocks: list[str] = Field(min_length=1)
    mode: UIAutomationMode = UIAutomationMode.COPILOT
    pause_between_blocks_ms: int | None = None
    focus_target: str | None = Field(default=None, validation_alias=AliasChoices("focus_target", "target_window"))
    ensure_window_contains: str | None = None
    approved: bool = False
    timeout_ms: int | None = None
    watchdog_timeout_ms: int | None = None


class MoveMouseRequest(JarvisBaseModel):
    x: int
    y: int
    duration_seconds: float | None = None
    relative: bool = False
    approved: bool = False


class ClickRequest(JarvisBaseModel):
    button: MouseButton = MouseButton.LEFT
    double: bool = False
    x: int | None = None
    y: int | None = None
    duration_seconds: float | None = None
    approved: bool = False


class ShortcutRequest(JarvisBaseModel):
    keys: tuple[str, ...] = ()
    approved: bool = False


class FocusWindowRequest(JarvisBaseModel):
    target: str
    approved: bool = False


class CloseWindowRequest(JarvisBaseModel):
    target: str | None = None
    approved: bool = False


class ClickVisualTargetRequest(JarvisBaseModel):
    label: str
    kind: str | None = None
    button: MouseButton = MouseButton.LEFT
    double: bool = False
    focus_target: str | None = Field(default=None, validation_alias=AliasChoices("focus_target", "target_window"))
    approved: bool = False


class CancellationRequest(JarvisBaseModel):
    correlation_id: str


class UIOperationReceipt(JarvisBaseModel):
    correlation_id: str
    operation_name: str
    risk_level: UIRiskLevel
    success: bool
    status: UIOperationStatus = UIOperationStatus.EXECUTED
    message: str
    active_window: WindowInfo | None = None
    confirmation_required: bool = False
    security_decision: str = "allowed"
    data: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
