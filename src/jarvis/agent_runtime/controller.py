from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4

from jarvis.action_executor import (
    ActionExecutor,
    ActionObservation,
    ActionRequest,
    ActionStatus,
)
from jarvis.models_runtime.base import ModelMessage, ModelRequest, ModelResponse

from .models import AgentRunRequest, AgentRunResult, AgentRunStatus


ModelInference = Callable[[ModelRequest], ModelResponse]
EventSink = Callable[[str, dict[str, Any]], None]


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
        action_executor: ActionExecutor,
        event_sink: EventSink | None = None,
        clock: Callable[[], float] = monotonic,
        max_repeated_signature: int = 2,
    ) -> None:
        self._infer_model = infer_model
        self._actions = action_executor
        self._event_sink = event_sink
        self._clock = clock
        self._max_repeated_signature = max(max_repeated_signature, 1)
        self._pending_confirmations: dict[str, _PendingConfirmationRun] = {}

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        run_id = request.run_id or f"agent-{uuid4().hex}"
        correlation_id = request.correlation_id or run_id
        deadline = self._clock() + request.timeout_seconds
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
                return terminal
            self._publish("model_requested", run_id, correlation_id, step_count)
            try:
                message_snapshot = [
                    message.model_copy(deep=True) for message in messages
                ]
                if base_request is None:
                    model_request = ModelRequest(
                        task_type="agent",
                        logical_model="general_assistant",
                        required_capabilities=("chat", "tool_calling"),
                        messages=message_snapshot,
                        tools=self._actions.tool_schemas(),
                        tool_choice="auto",
                        correlation_id=correlation_id,
                    )
                else:
                    capabilities = tuple(
                        dict.fromkeys(
                            (*base_request.required_capabilities, "tool_calling")
                        )
                    )
                    model_request = base_request.model_copy(
                        update={
                            "task_type": "agent",
                            "required_capabilities": capabilities,
                            "messages": message_snapshot,
                            "tools": self._merge_tool_schemas(
                                base_request.tools,
                                self._actions.tool_schemas(),
                            ),
                            "tool_choice": base_request.tool_choice or "auto",
                            "correlation_id": correlation_id,
                        }
                    )
                response = self._infer_model(model_request)
                last_response = response
            except Exception as exc:  # noqa: BLE001
                result = AgentRunResult(
                    run_id=run_id,
                    status=AgentRunStatus.FAILED,
                    step_count=step_count,
                    model_call_count=model_calls + 1,
                    observations=observations,
                    error_type=type(exc).__name__,
                    **self._response_metadata(last_response),
                )
                self._publish("failed", run_id, correlation_id, step_count)
                return result
            model_calls += 1
            if not response.tool_calls:
                result = AgentRunResult(
                    run_id=run_id,
                    status=AgentRunStatus.COMPLETED,
                    final_answer=response.content,
                    step_count=step_count,
                    model_call_count=model_calls,
                    observations=observations,
                    **self._response_metadata(response),
                )
                self._publish("completed", run_id, correlation_id, step_count)
                return result

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
                return result

            messages.append(
                ModelMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
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
                    return result
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
                        **self._response_metadata(response),
                    )
                    self._publish(
                        "waiting_confirmation",
                        run_id,
                        correlation_id,
                        step_count,
                    )
                    return result

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
                step_count=step_count,
                model_call_count=model_calls,
                observations=observations,
            )
        if self._clock() >= deadline:
            return AgentRunResult(
                run_id=run_id,
                status=AgentRunStatus.TIMED_OUT,
                step_count=step_count,
                model_call_count=model_calls,
                observations=observations,
            )
        return None

    @staticmethod
    def _response_metadata(response: ModelResponse | None) -> dict[str, str | None]:
        if response is None:
            return {}
        return {
            "provider_name": response.provider_name,
            "logical_model": response.logical_model,
            "model_name": response.model_name,
        }

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
        return json.dumps(
            observation.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

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
