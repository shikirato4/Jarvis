from __future__ import annotations

from datetime import datetime, UTC
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ProviderKind(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


class ModelMedia(JarvisBaseModel):
    file_id: str
    category: str
    mime_type: str
    safe_name: str
    data: bytes = Field(exclude=True, repr=False)


class ModelToolCall(JarvisBaseModel):
    id: str
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class ModelMessage(JarvisBaseModel):
    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class ModelRequest(JarvisBaseModel):
    task_type: str
    logical_model: str | None = None
    required_capabilities: tuple[str, ...] = ()
    messages: list[ModelMessage] = Field(default_factory=list)
    media: list[ModelMedia] = Field(default_factory=list, exclude=True, repr=False)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
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
    # Tool-only assistant turns legitimately carry no textual content.
    content: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    fallback_used: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    usage: ModelUsage = Field(default_factory=ModelUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(JarvisBaseModel):
    text: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    done: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(JarvisBaseModel):
    provider_name: str
    healthy: bool
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(Protocol):
    provider_name: str
    provider_kind: ProviderKind

    def health_check(self) -> ProviderHealth: ...

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse: ...

    def stream_infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None, cancel_check=None): ...
