from __future__ import annotations

from datetime import datetime, UTC

from pydantic import BaseModel, ConfigDict, Field


class JarvisBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class TimestampedModel(JarvisBaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
