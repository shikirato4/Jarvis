from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class ToolResult(JarvisBaseModel):
    status: ToolExecutionStatus = ToolExecutionStatus.SUCCESS
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ToolInvocationReceipt(JarvisBaseModel):
    correlation_id: str
    tool: str
    status: ToolExecutionStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
