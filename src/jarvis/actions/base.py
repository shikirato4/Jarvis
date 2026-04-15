from __future__ import annotations

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ActionPayload(JarvisBaseModel):
    metadata: dict[str, object] = Field(default_factory=dict)
