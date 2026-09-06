from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from jarvis.models.base import JarvisBaseModel
from jarvis.models_runtime.base import ModelMessage


class ContextBudgetError(RuntimeError):
    """The configured model window leaves no usable input budget."""


class ContextOverflowError(RuntimeError):
    """Protected context cannot fit without silently losing required data."""


class ContextSource(StrEnum):
    HISTORY = "history"
    TOOL = "tool"
    ATTACHMENT = "attachment"
    MEMORY = "memory"
    CURRENT = "current"


class ContextFragment(JarvisBaseModel):
    fragment_id: str
    source: ContextSource
    content: str
    score: float = 0.0
    critical: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class ConversationContextRequest(JarvisBaseModel):
    system_message: ModelMessage
    history: list[ModelMessage] = Field(default_factory=list)
    tool_observations: list[ModelMessage] = Field(default_factory=list)
    attachment_context: list[ContextFragment] = Field(default_factory=list)
    embedded_attachment_count: int = Field(default=0, ge=0)
    memory_candidates: list[ContextFragment] = Field(default_factory=list)
    current_message: ModelMessage
    max_context_tokens: int = Field(gt=0)
    reserve_output_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    correlation_id: str | None = None

    @model_validator(mode="after")
    def _validate_usable_budget(self) -> ConversationContextRequest:
        if self.usable_input_tokens <= 0:
            raise ContextBudgetError(
                "output reserve and safety margin consume the context window"
            )
        return self

    @property
    def usable_input_tokens(self) -> int:
        return (
            self.max_context_tokens
            - self.reserve_output_tokens
            - self.safety_margin_tokens
        )


class ConversationContextStats(JarvisBaseModel):
    history_message_count: int = 0
    tool_observation_count: int = 0
    attachment_context_count: int = 0
    memory_candidate_count: int = 0
    memory_selected_count: int = 0
    estimated_context_tokens: int = 0
    context_compacted: bool = False
    context_truncated: bool = False


class BuiltConversationContext(JarvisBaseModel):
    messages: list[ModelMessage]
    stats: ConversationContextStats
