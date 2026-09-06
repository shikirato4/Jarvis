from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

from jarvis.action_executor import ActionObservation
from jarvis.models.base import JarvisBaseModel
from jarvis.models_runtime.base import ModelMessage, ModelRequest


class AgentRunStatus(StrEnum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    INTERRUPTED = "interrupted"
    WAITING_CONFIRMATION = "waiting_confirmation"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    STEP_LIMIT = "step_limit"


class AgentOutcomeCode(StrEnum):
    """Stable public classification; never contains provider reasoning text."""

    NORMAL_STOP = "NORMAL_STOP"
    MODEL_OUTPUT_TRUNCATED = "MODEL_OUTPUT_TRUNCATED"
    TOOL_CALLS_PENDING = "TOOL_CALLS_PENDING"
    SECURITY_CONFIRMATION_REQUIRED = "SECURITY_CONFIRMATION_REQUIRED"
    SECURITY_DENIED = "SECURITY_DENIED"
    TOOL_ARGUMENT_ERROR = "TOOL_ARGUMENT_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    MODEL_CONTINUATION_FAILURE = "MODEL_CONTINUATION_FAILURE"
    PROVIDER_TRANSPORT_FAILURE = "PROVIDER_TRANSPORT_FAILURE"
    PROVIDER_PROCESS_EXIT = "PROVIDER_PROCESS_EXIT"
    USER_CANCELLED = "USER_CANCELLED"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"


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
    stream_callback: Callable[[str], None] | None = Field(
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
    outcome_code: AgentOutcomeCode | None = None
    provider_name: str | None = None
    logical_model: str | None = None
    model_name: str | None = None
    finish_reason: str | None = None
    stream_closed_normally: bool | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    requested_max_tokens: int | None = None
    provider_text_length: int | None = None
    chunk_count: int | None = None
    auto_continuation_count: int = 0
    model_request_count: int | None = None
    segment_finish_reasons: list[str | None] = Field(default_factory=list)
    segment_durations_seconds: list[float] = Field(default_factory=list)
    segment_completion_tokens: list[int] = Field(default_factory=list)
    total_completion_tokens: int | None = None
    total_response_token_limit: int | None = None
    deduplicated_overlap_chars: int = 0
    context_prevented_continuation: bool = False
    auto_continuation_failed: bool = False
    continuation_stop_reason: str | None = None
    original_finish_reason: str | None = None
    continuation_requested: bool = False
    assistant_final_present: bool = False
    context_messages: list[ModelMessage] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )
