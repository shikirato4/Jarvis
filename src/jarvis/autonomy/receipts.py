from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class AutonomyStepReceipt(JarvisBaseModel):
    mission_id: str
    step_id: str
    operation_name: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AutonomyMissionReceipt(JarvisBaseModel):
    mission_id: str
    status: str
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
