from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class WebSearchProviderStatus(JarvisBaseModel):
    provider: str = "disabled"
    enabled: bool = False
    available: bool = False
    configured: bool = False
    sends_private_context: bool = False
    max_results: int = 5
    timeout_seconds: float = 8.0
    message: str = ""


class WebSearchRequest(JarvisBaseModel):
    query: str
    max_results: int = 5


class WebSearchHit(JarvisBaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str = ""
    provider: str = "brave"
    rank: int = 0
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchResponse(JarvisBaseModel):
    status: str
    provider: str
    query: str
    hits: list[WebSearchHit] = Field(default_factory=list)
    message: str = ""
    searched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    used_private_context: bool = False
