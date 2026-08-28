from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

from jarvis.action_executor import ActionObservation
from jarvis.models.base import JarvisBaseModel
from jarvis.models_runtime.base import ModelMessage, ModelRequest


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"
    WAITING_CONFIRMATION = "waiting_confirmation"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STEP_LIMIT = "step_limit"


class AgentRunRequest(JarvisBaseModel):
    user_message: str = ""
    system_message: str = (
        "You are Jarvis. Use tools when needed, observe their results, and return "
        "only the final answer without revealing private chain-of-thought."
    )
    history: list[ModelMessage] = Field(default_factory=list)
    model_request: ModelRequest | None = Field(default=None, exclude=True, repr=False)
    max_steps: int = Field(default=8, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    run_id: str | None = None
    correlation_id: str | None = None
    source_surface: str | None = None
    cancellation_check: Callable[[], bool] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    prior_step_count: int = Field(default=0, ge=0, exclude=True, repr=False)
    prior_model_call_count: int = Field(default=0, ge=0, exclude=True, repr=False)
    prior_observations: list[ActionObservation] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )


class AgentRunResult(JarvisBaseModel):
    run_id: str
    status: AgentRunStatus
    final_answer: str | None = None
    step_count: int = 0
    model_call_count: int = 0
    observations: list[ActionObservation] = Field(default_factory=list)
    pending_confirmation: ActionObservation | None = None
    error_type: str | None = None
    provider_name: str | None = None
    logical_model: str | None = None
    model_name: str | None = None
