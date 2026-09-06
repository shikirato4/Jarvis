from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import monotonic
from typing import Any, Protocol
from uuid import uuid4

from jarvis.action_executor import (
    ActionExecutor,
    ActionObservation,
    ActionRequest,
    ActionStatus,
)
from jarvis.models_runtime.base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
    StreamChunk,
)

from .models import AgentOutcomeCode, AgentRunRequest, AgentRunResult, AgentRunStatus


ModelInference = Callable[[ModelRequest], ModelResponse]
EventSink = Callable[[str, dict[str, Any]], None]


class ModelStreamingInference(Protocol):
    def __call__(
        self,
        request: ModelRequest,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[StreamChunk]: ...


class ShadowObserver(Protocol):
    """Backend-neutral observer used to compare routing without controlling it."""

    def observe(
        self,
        *,
        text: str,
        selected_tools: list[str],
        success: bool,
        step_count: int,
        latency_ms: float,
        correction_required: bool,
    ) -> list[str]: ...


@dataclass(slots=True)
class _PendingConfirmationRun:
    request: AgentRunRequest
    model_request: ModelRequest
    messages: list[ModelMessage]
    observations: list[ActionObservation]
    model_calls: int
    step_count: int
    correlation_id: str
    action_request: ActionRequest
    confirmation_id: str
    deadline: float


class AgentController:
    """Bounded Nemotron tool loop; operational events never contain reasoning text."""

    def __init__(
        self,
        *,
        infer_model: ModelInference,
        stream_model: ModelStreamingInference | None = None,
        action_executor: ActionExecutor,
        event_sink: EventSink | None = None,
        shadow_observer: ShadowObserver | None = None,
        clock: Callable[[], float] = monotonic,
        max_repeated_signature: int = 2,
    ) -> None:
        self._infer_model = infer_model
        self._stream_model = stream_model
        self._actions = action_executor
        self._event_sink = event_sink
        self._shadow_observer = shadow_observer
        self._clock = clock
        self._max_repeated_signature = max(max_repeated_signature, 1)
        self._pending_confirmations: dict[str, _PendingConfirmationRun] = {}

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        started_at = self._clock()
        run_id = request.run_id or f"agent-{uuid4().hex}"
        correlation_id = request.correlation_id or run_id
        deadline = started_at + request.timeout_seconds
        base_request = request.model_request
        if base_request is None:
            messages = [
                ModelMessage(role="system", content=request.system_message),
                *request.history,
                ModelMessage(role="user", content=request.user_message),
            ]
        else:
            messages = [message.model_copy(deep=True) for message in base_request.messages]
        observations = [
            observation.model_copy(deep=True)
            for observation in request.prior_observations
        ]
        model_calls = request.prior_model_call_count
        step_count = request.prior_step_count
        last_response: ModelResponse | None = None
        signature_counts: dict[str, int] = {}
        structured_output_retries = 0
        provider_request_count = request.prior_model_call_count
        provider_finish_reasons: list[str | None] = []
        provider_durations_seconds: list[float] = []
        provider_completion_tokens: list[int] = []
        auto_continuation_count = 0
        deduplicated_overlap_chars = 0
        self._publish("started", run_id, correlation_id, step_count)

        while True:
            terminal = self._preflight_terminal(
                request=request,
                run_id=run_id,
                model_calls=model_calls,
                step_count=step_count,
                observations=observations,
                deadline=deadline,
            )
            if terminal is not None:
                suffix = (
                    "cancelled"
                    if terminal.status is AgentRunStatus.CANCELLED
                    else "timed_out"
                )
                self._publish(
                    suffix,
                    run_id,
                    correlation_id,
                    step_count,
                )
                return self._finish_with_shadow(request, terminal, started_at)
            if model_calls:
                self._publish(
                    "continuation_started",
                    run_id,
                    correlation_id,
                    step_count,
                )
            self._publish("model_requested", run_id, correlation_id, step_count)
            try:
                message_snapshot = [
                    message.model_copy(deep=True) for message in messages
                ]
                if base_request is None:
                    available_tools = self._actions.tool_schemas()
                    model_request = ModelRequest(
                        task_type="agent",
                        logical_model="general_assistant",
                        required_capabilities=("chat", "tool_calling"),
                        messages=self._with_capability_awareness(message_snapshot, available_tools),
                        tools=available_tools,
                        tool_choice="auto",
                        correlation_id=correlation_id,
                    )
                else:
                    available_tools = self._actions.tool_schemas()
                    capabilities = tuple(
                        dict.fromkeys(
                            (*base_request.required_capabilities, "tool_calling")
                        )
                    )
                    model_request = base_request.model_copy(
                        update={
                            "task_type": "agent",
                            "required_capabilities": capabilities,
                            "messages": self._with_capability_awareness(message_snapshot, available_tools),
                            "tools": self._merge_tool_schemas(
                                base_request.tools,
                                available_tools,
                            ),
                            "tool_choice": base_request.tool_choice or "auto",
                            "correlation_id": correlation_id,
                        }
                    )
                agent_phase = (
                    "agent_final"
                    if observations
                    else "structured_tool_call"
                    if model_request.metadata.get("agent_tool_required")
                    else "agent_initial"
                )
                model_request = model_request.model_copy(
                    update={
                        "metadata": {
                            **model_request.metadata,
                            "agent_phase": agent_phase,
                            "adaptive_output_budget": True,
                            "auto_continue_text": agent_phase == "agent_final",
                        }
                    }
                )
                response = self._normalize_model_response(
                    self._request_model(model_request, request),
                    model_request,
                )
                response_metadata = dict(response.metadata)
                request_count = int(
                    response_metadata.get("model_request_count") or 1
                )
                provider_request_count += request_count
                response_reasons = list(
                    response_metadata.get("segment_finish_reasons") or []
                )
                if not response_reasons:
                    response_reasons = [response_metadata.get("finish_reason")]
                provider_finish_reasons.extend(response_reasons)
                response_durations = list(
                    response_metadata.get("segment_durations_seconds") or []
                )
                if not response_durations:
                    response_durations = [round(response.latency_ms / 1000, 3)]
                provider_durations_seconds.extend(response_durations)
                response_tokens = list(
                    response_metadata.get("segment_completion_tokens") or []
                )
                if not response_tokens:
                    completion_tokens = response_metadata.get("completion_tokens")
                    if isinstance(completion_tokens, int):
                        response_tokens = [completion_tokens]
                provider_completion_tokens.extend(
                    int(value) for value in response_tokens if isinstance(value, int)
                )
                auto_continuation_count += int(
                    response_metadata.get("auto_continuation_count") or 0
                )
                deduplicated_overlap_chars += int(
                    response_metadata.get("deduplicated_overlap_chars") or 0
                )
                response_metadata.update(
                    {
                        "model_request_count": provider_request_count,
                        "segment_finish_reasons": provider_finish_reasons,
                        "segment_durations_seconds": provider_durations_seconds,
                        "segment_completion_tokens": provider_completion_tokens,
                        "total_completion_tokens": (
                            sum(provider_completion_tokens)
                            if provider_completion_tokens
                            else None
                        ),
                        "auto_continuation_count": auto_continuation_count,
                        "deduplicated_overlap_chars": deduplicated_overlap_chars,
                    }
                )
                response = response.model_copy(update={"metadata": response_metadata})
                last_response = response
            except Exception as exc:  # noqa: BLE001
                structured_retry_failed = bool(
                    structured_output_retries and not observations
                )
                outcome_code = (
                    AgentOutcomeCode.INVALID_MODEL_OUTPUT
                    if structured_retry_failed
                    else self._exception_outcome(
                        exc,
                        continuation=bool(observations),
                    )
                )
                result = AgentRunResult(
                    run_id=run_id,
                    status=AgentRunStatus.FAILED,
                    step_count=step_count,
                    model_call_count=model_calls + 1,
                    observations=observations,
                    error_type=(
                        "StructuredRetryFailure"
                        if structured_retry_failed
                        else type(exc).__name__
                    ),
                    outcome_code=outcome_code,
                    continuation_requested=bool(observations),
                    assistant_final_present=False,
                    **self._response_metadata(last_response),
                )
                self._publish(
                    "failed",
                    run_id,
                    correlation_id,
                    step_count,
                    outcome_code=outcome_code.value,
                    error_type=result.error_type,
                )
                return self._finish_with_shadow(request, result, started_at)
            model_calls += 1
            if (
                response.metadata.get("finish_reason") == "length"
                and not observations
                and model_request.metadata.get("agent_phase")
                == "structured_tool_call"
            ):
                if structured_output_retries < 1:
                    structured_output_retries += 1
                    self._publish(
                        "structured_retry_started",
                        run_id,
                        correlation_id,
                        step_count,
                        outcome_code=AgentOutcomeCode.INVALID_MODEL_OUTPUT.value,
                    )
                    messages.append(
                        ModelMessage(
                            role="system",
                            content=(
                                "The previous structured tool-call generation was "
                                "incomplete. Emit one complete, concise tool call "
                                "with valid arguments. Do not emit partial JSON or "
                                "explanatory prose."
                            ),
                        )
                    )
                    continue
                result = AgentRunResult(
                    run_id=run_id,
                    status=AgentRunStatus.FAILED,
                    final_answer=response.content or None,
                    step_count=step_count,
                    model_call_count=model_calls,
                    observations=observations,
                    error_type="TruncatedStructuredToolCall",
                    outcome_code=AgentOutcomeCode.INVALID_MODEL_OUTPUT,
                    continuation_requested=True,
                    assistant_final_present=False,
                    **self._response_metadata(response),
                )
                self._publish(
                    "failed",
                    run_id,
                    correlation_id,
                    step_count,
                    outcome_code=result.outcome_code.value,
                    error_type=result.error_type,
                )
                return self._finish_with_shadow(request, result, started_at)
            completion_error = self._completion_error(response)
            if completion_error is not None:
                truncated = response.metadata.get("finish_reason") == "length"
                outcome_code = (
                    AgentOutcomeCode.MODEL_OUTPUT_TRUNCATED
                    if truncated
                    else AgentOutcomeCode.MODEL_CONTINUATION_FAILURE
                    if observations
                    else AgentOutcomeCode.PROVIDER_TRANSPORT_FAILURE
                )
                result = AgentRunResult(
                    run_id=run_id,
                    status=(
                        AgentRunStatus.TRUNCATED
                        if truncated
                        else AgentRunStatus.INTERRUPTED
                    ),
                    final_answer=response.content or None,
                    step_count=step_count,
                    model_call_count=model_calls,
                    observations=observations,
                    error_type=completion_error,
                    outcome_code=outcome_code,
                    continuation_requested=bool(observations),
                    assistant_final_present=bool(response.content.strip()),
                    **self._response_metadata(response),
                )
                self._publish(
                    "output_truncated" if truncated else "interrupted",
                    run_id,
                    correlation_id,
                    step_count,
                    outcome_code=outcome_code.value,
                    finish_reason=response.metadata.get("finish_reason"),
                )
                return self._finish_with_shadow(request, result, started_at)
            if not response.tool_calls:
                tool_required = bool(
                    base_request is not None
                    and base_request.metadata.get("agent_tool_required")
                )
                if tool_required and not observations and model_calls == 1:
                    self._publish(
                        "tool_selection_retry",
                        run_id,
                        correlation_id,
                        step_count,
                    )
                    messages.append(
                        ModelMessage(
                            role="system",
                            content=(
                                "This is an operational request and requires an "
                                "observable tool action. Call one appropriate "
                                "available tool now; do not replace the action "
                                "with instructions or claim it was performed."
                            ),
                        )
                    )
                    continue
                if tool_required and not observations:
                    result = AgentRunResult(
                        run_id=run_id,
                        status=AgentRunStatus.FAILED,
                        final_answer=response.content.strip() or None,
                        step_count=step_count,
                        model_call_count=model_calls,
                        observations=observations,
                        error_type="MissingRequiredToolCall",
                        outcome_code=AgentOutcomeCode.INVALID_MODEL_OUTPUT,
                        continuation_requested=True,
                        assistant_final_present=bool(response.content.strip()),
                        **self._response_metadata(response),
                    )
                    self._publish(
                        "failed",
                        run_id,
                        correlation_id,
                        step_count,
                        outcome_code=result.outcome_code.value,
                        error_type=result.error_type,
                    )
                    return self._finish_with_shadow(request, result, started_at)
                final_content = self._ground_file_paths(
                    response.content,
                    observations,
                )
                final_answer = final_content.strip()
                if not final_answer:
                    outcome_code = (
                        AgentOutcomeCode.MODEL_CONTINUATION_FAILURE
                        if observations
                        else AgentOutcomeCode.INVALID_MODEL_OUTPUT
                    )
                    result = AgentRunResult(
                        run_id=run_id,
                        status=AgentRunStatus.FAILED,
                        step_count=step_count,
                        model_call_count=model_calls,
                        observations=observations,
                        error_type=(
                            "EmptyContinuationFinal"
                            if observations
                            else "EmptyModelResponse"
                        ),
                        outcome_code=outcome_code,
                        continuation_requested=bool(observations),
                        assistant_final_present=False,
                        **self._response_metadata(response),
                    )
                    self._publish(
                        "failed",
                        run_id,
                        correlation_id,
                        step_count,
                        outcome_code=outcome_code.value,
                        error_type=result.error_type,
                    )
                    return self._finish_with_shadow(request, result, started_at)
                outcome_code = self._terminal_outcome(observations)
                terminal_status = (
                    AgentRunStatus.COMPLETED
                    if outcome_code is AgentOutcomeCode.NORMAL_STOP
                    else AgentRunStatus.CANCELLED
                    if outcome_code is AgentOutcomeCode.USER_CANCELLED
                    else AgentRunStatus.BLOCKED
                    if outcome_code is AgentOutcomeCode.SECURITY_DENIED
                    else AgentRunStatus.FAILED
                )
                result = AgentRunResult(
                    run_id=run_id,
                    status=terminal_status,
                    final_answer=final_content,
                    step_count=step_count,
                    model_call_count=model_calls,
                    observations=observations,
                    outcome_code=outcome_code,
                    continuation_requested=bool(observations),
                    assistant_final_present=True,
                    context_messages=self._run_context_messages(
                        messages,
                        run_id,
                        response.model_copy(update={"content": final_content}),
                    ),
                    **self._response_metadata(response),
                )
                self._publish(
                    "completed"
                    if terminal_status is AgentRunStatus.COMPLETED
                    else "blocked"
                    if terminal_status is AgentRunStatus.BLOCKED
                    else "cancelled"
                    if terminal_status is AgentRunStatus.CANCELLED
                    else "failed",
                    run_id,
                    correlation_id,
                    step_count,
                    outcome_code=outcome_code.value,
                )
                return self._finish_with_shadow(request, result, started_at)

            remaining_steps = request.max_steps - step_count
            if remaining_steps <= 0 or len(response.tool_calls) > remaining_steps:
                result = AgentRunResult(
                    run_id=run_id,
                    status=AgentRunStatus.STEP_LIMIT,
                    step_count=step_count,
                    model_call_count=model_calls,
                    observations=observations,
                )
                self._publish("step_limit", run_id, correlation_id, step_count)
                return self._finish_with_shadow(request, result, started_at)

            messages.append(
                ModelMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    metadata={"agent_run_id": run_id},
                )
            )
            self._publish(
                "tool_calls_pending",
                run_id,
                correlation_id,
                step_count,
                outcome_code=AgentOutcomeCode.TOOL_CALLS_PENDING.value,
                tool_call_count=len(response.tool_calls),
            )
            for call in response.tool_calls:
                if step_count >= request.max_steps:
                    break
                signature = self._tool_signature(call.name, call.arguments)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
                if signature_counts[signature] > self._max_repeated_signature:
                    result = AgentRunResult(
                        run_id=run_id,
                        status=AgentRunStatus.BLOCKED,
                        step_count=step_count,
                        model_call_count=model_calls,
                        observations=observations,
                        error_type="RepeatedActionLoop",
                        **self._response_metadata(response),
                    )
                    self._publish("blocked", run_id, correlation_id, step_count)
                    return self._finish_with_shadow(request, result, started_at)
                self._publish(
                    "tool_requested",
                    run_id,
                    correlation_id,
                    step_count,
                    call_id=call.id,
                    action=call.name,
                )
                action_request = ActionRequest(
                    run_id=run_id,
                    call_id=call.id,
                    action=call.name,
                    arguments=(
                        call.arguments
                        if isinstance(call.arguments, dict)
                        else {"raw_arguments": call.arguments}
                    ),
                    correlation_id=correlation_id,
                    metadata={"source_surface": request.source_surface},
                )
                observation = self._actions.execute(action_request)
                observations.append(observation)
                step_count += 1
                self._publish(
                    "tool_observed",
                    run_id,
                    correlation_id,
                    step_count,
                    call_id=call.id,
                    action=call.name,
                    status=observation.status.value,
                )
                messages.append(
                    ModelMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=self._observation_content(observation),
                        metadata={"agent_run_id": run_id},
                    )
                )
                if observation.status is ActionStatus.CONFIRMATION_REQUIRED:
                    assert observation.confirmation_id is not None
                    self._pending_confirmations[run_id] = _PendingConfirmationRun(
                        request=request,
                        model_request=model_request,
                        messages=[
                            message.model_copy(deep=True) for message in messages[:-1]
                        ],
                        observations=[
                            item.model_copy(deep=True) for item in observations[:-1]
                        ],
                        model_calls=model_calls,
                        step_count=step_count,
                        correlation_id=correlation_id,
                        action_request=action_request,
                        confirmation_id=observation.confirmation_id,
                        deadline=deadline,
                    )
                    result = AgentRunResult(
                        run_id=run_id,
                        status=AgentRunStatus.WAITING_CONFIRMATION,
                        step_count=step_count,
                        model_call_count=model_calls,
                        observations=observations,
                        pending_confirmation=observation,
                        outcome_code=AgentOutcomeCode.SECURITY_CONFIRMATION_REQUIRED,
                        continuation_requested=False,
                        assistant_final_present=False,
                        **self._response_metadata(response),
                    )
                    self._publish(
                        "waiting_confirmation",
                        run_id,
                        correlation_id,
                        step_count,
                        outcome_code=(
                            AgentOutcomeCode.SECURITY_CONFIRMATION_REQUIRED.value
                        ),
                    )
                    return result

    def _finish_with_shadow(
        self,
        request: AgentRunRequest,
        result: AgentRunResult,
        started_at: float,
    ) -> AgentRunResult:
        if self._shadow_observer is None:
            return result
        selected_tools = list(
            dict.fromkeys(observation.action for observation in result.observations)
        )
        payload = {
            "run_id": result.run_id,
            "correlation_id": request.correlation_id or result.run_id,
            "step_count": result.step_count,
            "selected_tool_count": len(selected_tools),
        }
        if self._event_sink is not None:
            self._event_sink("neural_router.started", payload)
        try:
            self._shadow_observer.observe(
                text=request.user_message,
                selected_tools=selected_tools,
                success=result.status is AgentRunStatus.COMPLETED,
                step_count=result.step_count,
                latency_ms=max((self._clock() - started_at) * 1_000, 0.0),
                correction_required=result.status is not AgentRunStatus.COMPLETED,
            )
        except Exception as exc:  # noqa: BLE001
            if self._event_sink is not None:
                self._event_sink(
                    "neural_router.failed",
                    {**payload, "status": "failed", "error_type": type(exc).__name__},
                )
            return result
        if self._event_sink is not None:
            self._event_sink(
                "neural_router.completed",
                {**payload, "status": "success"},
            )
        return result

    def _request_model(
        self,
        model_request: ModelRequest,
        run_request: AgentRunRequest,
    ) -> ModelResponse:
        if self._stream_model is None or run_request.stream_callback is None:
            return self._infer_model(model_request)
        started_at = self._clock()
        content_parts: list[str] = []
        tool_calls_by_id: dict[str, ModelToolCall] = {}
        metadata: dict[str, Any] = {"completion_contract_enforced": True}
        terminal_seen = False
        for chunk in self._stream_model(
            model_request,
            cancel_check=run_request.cancellation_check,
        ):
            metadata.update(chunk.metadata)
            terminal_seen = terminal_seen or chunk.done
            if chunk.error:
                raise RuntimeError(chunk.error)
            if chunk.text:
                content_parts.append(chunk.text)
                run_request.stream_callback(chunk.text)
            for tool_call in chunk.tool_calls:
                tool_calls_by_id[tool_call.id] = tool_call
        if not terminal_seen:
            metadata.update(
                {
                    "stream_closed_normally": False,
                    "incomplete": True,
                    "finish_reason": None,
                }
            )
        metadata["provider_text_length"] = len("".join(content_parts))
        provider_name = str(
            metadata.get("provider")
            or metadata.get("active_provider")
            or "nemotron-stream"
        )
        model_name = str(
            metadata.get("model")
            or metadata.get("active_model")
            or model_request.logical_model
            or "nemotron"
        )
        return ModelResponse(
            provider_name=provider_name,
            provider_kind=(
                ProviderKind.REMOTE
                if provider_name == "nvidia_nim"
                else ProviderKind.LOCAL
            ),
            logical_model=model_request.logical_model or "general_assistant",
            model_name=model_name,
            content="".join(content_parts),
            tool_calls=list(tool_calls_by_id.values()),
            latency_ms=max((self._clock() - started_at) * 1_000, 0.0),
            metadata=metadata,
        )

    @staticmethod
    def _normalize_model_response(
        response: ModelResponse | Any,
        request: ModelRequest,
    ) -> ModelResponse:
        """Accept legacy in-process doubles while keeping one runtime contract."""

        if isinstance(response, ModelResponse):
            return response
        provider_name = str(getattr(response, "provider_name", "in_process_model"))
        return ModelResponse(
            provider_name=provider_name,
            provider_kind=(
                ProviderKind.REMOTE
                if provider_name == "nvidia_nim"
                else ProviderKind.LOCAL
            ),
            logical_model=str(
                getattr(response, "logical_model", None)
                or request.logical_model
                or "general_assistant"
            ),
            model_name=str(
                getattr(response, "model_name", None)
                or request.logical_model
                or "in_process_model"
            ),
            content=str(getattr(response, "content", "") or ""),
            tool_calls=list(getattr(response, "tool_calls", []) or []),
            latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
            fallback_used=bool(getattr(response, "fallback_used", False)),
            metadata=dict(getattr(response, "metadata", {}) or {}),
        )

    @staticmethod
    def _run_context_messages(
        messages: list[ModelMessage],
        run_id: str,
        final_response: ModelResponse,
    ) -> list[ModelMessage]:
        operational = [
            message.model_copy(deep=True)
            for message in messages
            if message.metadata.get("agent_run_id") == run_id
        ]
        operational.append(
            ModelMessage(
                role="assistant",
                content=final_response.content,
                metadata={"agent_run_id": run_id},
            )
        )
        return operational

    def resume_confirmation(
        self,
        run_id: str,
        confirmation_id: str,
    ) -> AgentRunResult:
        pending = self._pending_confirmations.get(run_id)
        if pending is None or pending.confirmation_id != confirmation_id:
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.BLOCKED,
                error_type="PendingConfirmationNotFound",
            )
        self._pending_confirmations.pop(run_id, None)
        if not self._actions.approve_confirmation(confirmation_id):
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.BLOCKED,
                step_count=pending.step_count,
                model_call_count=pending.model_calls,
                observations=pending.observations,
                error_type="ConfirmationApprovalRejected",
            )
        observation = self._actions.execute(
            pending.action_request.model_copy(
                update={"confirmation_token": confirmation_id}
            )
        )
        observations = [*pending.observations, observation]
        messages = [
            *pending.messages,
            ModelMessage(
                role="tool",
                name=pending.action_request.action,
                tool_call_id=pending.action_request.call_id,
                content=self._observation_content(observation),
            ),
        ]
        remaining_timeout = pending.deadline - self._clock()
        if remaining_timeout <= 0:
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.TIMED_OUT,
                step_count=pending.step_count,
                model_call_count=pending.model_calls,
                observations=observations,
            )
        self._publish(
            "resumed",
            run_id,
            pending.correlation_id,
            pending.step_count,
        )
        continuation_request = pending.model_request.model_copy(
            update={"messages": messages}
        )
        return self.run(
            AgentRunRequest(
                user_message=pending.request.user_message,
                system_message=pending.request.system_message,
                model_request=continuation_request,
                max_steps=pending.request.max_steps,
                timeout_seconds=remaining_timeout,
                run_id=run_id,
                correlation_id=pending.correlation_id,
                cancellation_check=pending.request.cancellation_check,
                stream_callback=pending.request.stream_callback,
                source_surface=pending.request.source_surface,
                prior_step_count=pending.step_count,
                prior_model_call_count=pending.model_calls,
                prior_observations=observations,
            )
        )

    def cancel_pending_confirmation(self, run_id: str) -> bool:
        return self._pending_confirmations.pop(run_id, None) is not None

    def _preflight_terminal(
        self,
        *,
        request: AgentRunRequest,
        run_id: str,
        model_calls: int,
        step_count: int,
        observations: list,
        deadline: float,
    ) -> AgentRunResult | None:
        if request.cancellation_check is not None and request.cancellation_check():
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.CANCELLED,
                outcome_code=AgentOutcomeCode.USER_CANCELLED,
                step_count=step_count,
                model_call_count=model_calls,
                observations=observations,
            )
        if self._clock() >= deadline:
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.TIMED_OUT,
                outcome_code=(
                    AgentOutcomeCode.MODEL_CONTINUATION_FAILURE
                    if observations
                    else AgentOutcomeCode.PROVIDER_TRANSPORT_FAILURE
                ),
                step_count=step_count,
                model_call_count=model_calls,
                observations=observations,
            )
        return None

    @staticmethod
    def _response_metadata(response: ModelResponse | None) -> dict[str, Any]:
        if response is None:
            return {}
        return {
            "provider_name": response.provider_name,
            "logical_model": response.logical_model,
            "model_name": response.model_name,
            "finish_reason": response.metadata.get("finish_reason"),
            "stream_closed_normally": response.metadata.get(
                "stream_closed_normally"
            ),
            "prompt_tokens": response.metadata.get("prompt_tokens")
            or response.usage.input_tokens,
            "completion_tokens": response.metadata.get("completion_tokens")
            or response.usage.output_tokens,
            "requested_max_tokens": response.metadata.get(
                "requested_max_tokens"
            ),
            "provider_text_length": response.metadata.get(
                "provider_text_length", len(response.content)
            ),
            "chunk_count": response.metadata.get("chunks"),
            "auto_continuation_count": int(
                response.metadata.get("auto_continuation_count") or 0
            ),
            "model_request_count": response.metadata.get("model_request_count"),
            "segment_finish_reasons": list(
                response.metadata.get("segment_finish_reasons") or []
            ),
            "segment_durations_seconds": list(
                response.metadata.get("segment_durations_seconds") or []
            ),
            "segment_completion_tokens": list(
                response.metadata.get("segment_completion_tokens") or []
            ),
            "total_completion_tokens": response.metadata.get(
                "total_completion_tokens"
            ),
            "total_response_token_limit": response.metadata.get(
                "total_response_token_limit"
            ),
            "deduplicated_overlap_chars": int(
                response.metadata.get("deduplicated_overlap_chars") or 0
            ),
            "context_prevented_continuation": bool(
                response.metadata.get("context_prevented_continuation")
            ),
            "auto_continuation_failed": bool(
                response.metadata.get("auto_continuation_failed")
            ),
            "continuation_stop_reason": response.metadata.get(
                "continuation_stop_reason"
            ),
            "original_finish_reason": response.metadata.get(
                "original_finish_reason"
            ),
        }

    @staticmethod
    def _completion_error(response: ModelResponse) -> str | None:
        """Reject transport/semantic terminal states that are not authoritative."""
        metadata = response.metadata
        if not metadata.get("completion_contract_enforced"):
            # Synthetic/in-process providers predate the transport contract.
            return None
        if metadata.get("stream_closed_normally") is not True:
            return "AbnormalStreamTermination"
        finish_reason = metadata.get("finish_reason")
        if finish_reason == "length":
            return "OutputTokenLimitReached"
        if finish_reason is None:
            return "MissingFinishReason"
        expected = "tool_calls" if response.tool_calls else "stop"
        if finish_reason != expected:
            return f"UnexpectedFinishReason:{finish_reason}"
        if metadata.get("incomplete"):
            return "IncompleteModelResponse"
        return None

    @staticmethod
    def _exception_outcome(
        exc: BaseException,
        *,
        continuation: bool,
    ) -> AgentOutcomeCode:
        if continuation:
            return AgentOutcomeCode.MODEL_CONTINUATION_FAILURE
        lowered = f"{type(exc).__name__}: {exc}".casefold()
        if any(
            marker in lowered
            for marker in (
                "process exited",
                "process exit",
                "server exited",
                "runtime exited",
                "broken pipe",
            )
        ):
            return AgentOutcomeCode.PROVIDER_PROCESS_EXIT
        return AgentOutcomeCode.PROVIDER_TRANSPORT_FAILURE

    @staticmethod
    def _terminal_outcome(
        observations: list[ActionObservation],
    ) -> AgentOutcomeCode:
        if not observations:
            return AgentOutcomeCode.NORMAL_STOP
        last = observations[-1]
        if last.status is ActionStatus.SUCCESS:
            return AgentOutcomeCode.NORMAL_STOP
        if last.status is ActionStatus.BLOCKED:
            return AgentOutcomeCode.SECURITY_DENIED
        if last.status is ActionStatus.CANCELLED:
            return AgentOutcomeCode.USER_CANCELLED
        error = str(last.error_type or last.data.get("error_code") or "").casefold()
        if "validation" in error or "argument" in error:
            return AgentOutcomeCode.TOOL_ARGUMENT_ERROR
        if "timeout" in error or "timed_out" in error:
            return AgentOutcomeCode.TOOL_TIMEOUT
        if "notfound" in error or "not_found" in error or "unavailable" in error:
            return AgentOutcomeCode.CAPABILITY_UNAVAILABLE
        return AgentOutcomeCode.TOOL_EXECUTION_ERROR

    @staticmethod
    def _with_capability_awareness(
        messages: list[ModelMessage], tools: list[dict[str, Any]]
    ) -> list[ModelMessage]:
        """Describe exposed local capabilities from the same registry as tools."""
        names = [
            str(schema.get("function", {}).get("name"))
            for schema in tools
            if isinstance(schema, dict)
            and isinstance(schema.get("function"), dict)
            and schema["function"].get("name")
        ]
        if not names:
            return messages
        namespaces = sorted({name.split(".", 1)[0] for name in names})
        terminal_names = [name for name in names if name.startswith("terminal.")]
        instruction = (
            "Capacidades JARVIS disponibles en esta solicitud, derivadas del "
            f"registro real de tools: {', '.join(namespaces)}."
        )
        if terminal_names:
            instruction += (
                " TerminalRuntime está disponible mediante: "
                f"{', '.join(terminal_names)}."
            )
        instruction += (
            " Si el usuario solicita explícitamente ejecutar una acción y una "
            "tool registrada puede hacerla, usa esa tool en lugar de indicarle "
            "que la haga manualmente; las preguntas informativas no obligan a "
            "usar tools. Después de recibir una observación de tool, responde "
            "con una respuesta final concisa y basada en esa observación; nunca "
            "termines el turno con contenido visible vacío. Toda acción sigue "
            "pasando por SecurityGate."
        )
        instruction += (
            " Para localizar un archivo o carpeta por nombre, comienza con "
            "files.search y conserva el alcance real de la observación. Nunca "
            "afirmes ni sugieras como resultado una ruta que no aparezca en una "
            "observación; si el alcance fue insuficiente, usa otra capacidad "
            "permitida o explica esa limitación."
        )
        updated = [message.model_copy(deep=True) for message in messages]
        system_index = next(
            (index for index, message in enumerate(updated) if message.role == "system"),
            None,
        )
        if system_index is None:
            updated.insert(0, ModelMessage(role="system", content=instruction))
        elif instruction not in updated[system_index].content:
            updated[system_index].content = (
                f"{updated[system_index].content}\n{instruction}"
            )
        return updated

    @staticmethod
    def _ground_file_paths(
        content: str,
        observations: list[ActionObservation],
    ) -> str:
        file_observations = [
            observation
            for observation in observations
            if observation.action.startswith("files.")
        ]
        if not file_observations:
            return content
        observed = json.dumps(
            [observation.data for observation in file_observations],
            ensure_ascii=False,
            default=str,
        ).casefold()
        removed = False

        def replace_code_path(match: re.Match[str]) -> str:
            nonlocal removed
            candidate = match.group(1).strip()
            if candidate.casefold() in observed:
                return match.group(0)
            removed = True
            return "`[ruta no verificada omitida]`"

        grounded = re.sub(
            r"`([^`\r\n]*(?:[A-Za-z]:[\\/]|\\\\)[^`\r\n]*)`",
            replace_code_path,
            content,
        )
        if removed:
            grounded = (
                f"{grounded.rstrip()}\n\n"
                "Omití una ruta no verificada porque no apareció en las "
                "observaciones de las capacidades ejecutadas."
            )
        return grounded

    @staticmethod
    def _tool_signature(name: str, arguments: dict[str, Any] | str) -> str:
        value = json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _merge_tool_schemas(
        existing: list[dict[str, Any]],
        available: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        names: set[str] = set()
        for schema in [*existing, *available]:
            function = schema.get("function") if isinstance(schema, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or name in names:
                continue
            names.add(name)
            merged.append(schema)
        return merged

    @staticmethod
    def _observation_content(observation) -> str:
        data = observation.data
        if observation.action.startswith("files."):
            data = AgentController._compact_file_observation(data)
        payload = {
            "action": observation.action,
            "status": observation.status.value,
            "message": observation.message,
            "data": data,
        }
        if observation.error_type:
            payload["error_type"] = observation.error_type
        if observation.confirmation_id:
            payload["confirmation_id"] = observation.confirmation_id
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _compact_file_observation(data: dict[str, Any]) -> dict[str, Any]:
        """Keep model-critical file facts while bounding 4K continuation input."""
        compact = {
            key: data[key]
            for key in (
                "operation",
                "source",
                "destination",
                "status",
                "verified",
                "error_code",
                "path",
                "truncated",
                "scanned",
                "bytes_read",
            )
            if key in data
        }
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            compact["metadata"] = {
                key: metadata[key]
                for key in (
                    "name",
                    "path",
                    "exists",
                    "type",
                    "is_file",
                    "is_directory",
                    "size",
                    "extension",
                    "mime_type",
                )
                if key in metadata
            }
        content = data.get("content")
        if isinstance(content, str):
            compact["content"] = content[:2_000]
            if len(content) > 2_000:
                compact["content_truncated_for_model"] = True
                compact["content_original_chars"] = len(content)
        items = data.get("items")
        if isinstance(items, list):
            # Directory listings can contain dozens of rich metadata fields.
            # Preserve only facts needed for the continuation so a successful
            # files.list cannot overflow the compact 4K request budget.
            compact["items"] = [
                {
                    key: item[key]
                    for key in (
                        "name",
                        "path",
                        "exists",
                        "type",
                        "is_file",
                        "is_directory",
                        "size",
                    )
                    if isinstance(item, dict) and key in item
                }
                for item in items[:20]
            ]
            if len(items) > 20:
                compact["items_truncated_for_model"] = True
                compact["item_count"] = len(items)
        return compact

    def _publish(
        self,
        suffix: str,
        run_id: str,
        correlation_id: str,
        step_count: int,
        **extra: Any,
    ) -> None:
        if self._event_sink is None:
            return
        self._event_sink(
            f"agent_controller.{suffix}",
            {
                "run_id": run_id,
                "correlation_id": correlation_id,
                "step_count": step_count,
                **extra,
            },
        )
