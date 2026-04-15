from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ActionResult(JarvisBaseModel):
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ActionStep(JarvisBaseModel):
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ActionExecutionReceipt(JarvisBaseModel):
    correlation_id: str
    action: str
    status: ExecutionStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rollback_attempted: bool = False
    rollback_succeeded: bool = False
