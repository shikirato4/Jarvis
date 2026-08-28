from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ActionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ActionRequest(JarvisBaseModel):
    run_id: str
    call_id: str
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    confirmation_token: str | None = Field(default=None, exclude=True, repr=False)
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class ActionObservation(JarvisBaseModel):
    run_id: str
    call_id: str
    action: str
    status: ActionStatus
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    error_type: str | None = None
    confirmation_id: str | None = None
    deduplicated: bool = False
    latency_ms: float = 0.0
