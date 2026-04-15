from __future__ import annotations

from typing import Any

from pydantic import Field

from jarvis.actions.models import ActionExecutionReceipt, ActionStep
from jarvis.models.base import JarvisBaseModel


class OrchestrationRequest(JarvisBaseModel):
    intent: str | None = None
    query: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    plan: list[ActionStep] = Field(default_factory=list)
    persist_input: bool = True


class OrchestrationResponse(JarvisBaseModel):
    correlation_id: str
    resolved_intent: str
    plan: list[ActionStep] = Field(default_factory=list)
    receipts: list[ActionExecutionReceipt] = Field(default_factory=list)
