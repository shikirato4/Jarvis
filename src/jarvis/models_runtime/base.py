from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ProviderKind(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class ModelMessage(JarvisBaseModel):
    role: str
    content: str


class ModelRequest(JarvisBaseModel):
    task_type: str
    logical_model: str | None = None
    required_capabilities: tuple[str, ...] = ()
    messages: list[ModelMessage] = Field(default_factory=list)
    prompt: str | None = None
    temperature: float | None = None
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None


class ModelUsage(JarvisBaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ModelResponse(JarvisBaseModel):
    provider_name: str
    provider_kind: ProviderKind
    logical_model: str
    model_name: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    fallback_used: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    usage: ModelUsage = Field(default_factory=ModelUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(JarvisBaseModel):
    text: str = ""
    done: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(JarvisBaseModel):
    provider_name: str
    healthy: bool
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(Protocol):
    provider_name: str
    provider_kind: ProviderKind

    def health_check(self) -> ProviderHealth: ...

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse: ...

    def stream_infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None, cancel_check=None): ...
