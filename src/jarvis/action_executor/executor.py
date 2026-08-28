from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from jarvis.actions.models import ExecutionStatus
from jarvis.actions.router import ActionRouter
from jarvis.core.errors import ActionExecutionError

from .models import ActionObservation, ActionRequest, ActionStatus
from .security import SecurityDecisionStatus, TypedActionSecurityGate


EventSink = Callable[[str, dict[str, Any]], None]


class ActionExecutor:
    """Typed, security-gated adapter over the existing action registry."""

    def __init__(
        self,
        router: ActionRouter,
        *,
        security_gate: TypedActionSecurityGate | None = None,
        event_sink: EventSink | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._router = router
        self._security_gate = security_gate or TypedActionSecurityGate()
        self._event_sink = event_sink
        self._clock = clock
        self._completed: dict[tuple[str, str], ActionObservation] = {}
        self._idempotency: dict[tuple[str, str], ActionObservation] = {}

    def approve_confirmation(self, confirmation_id: str) -> bool:
        return self._security_gate.approve(confirmation_id)

    def execute(self, request: ActionRequest) -> ActionObservation:
        call_key = (request.run_id, request.call_id)
        existing = self._completed.get(call_key)
        if existing is not None:
            observation = existing.model_copy(update={"deduplicated": True})
            self._publish("deduplicated", request, observation.status)
            return observation
        definition = self._router.registry.get(request.action)
        if definition is None:
            return self._failure(request, "ActionNotFoundError")
        try:
            payload = definition.payload_model.model_validate(request.arguments)
        except ValidationError:
            return self._failure(request, "ActionValidationError")
        normalized_request = request.model_copy(
            update={"arguments": payload.model_dump(mode="json")}
        )
        security = self._security_gate.evaluate(definition, normalized_request)
        if security.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED:
            observation = ActionObservation(
                run_id=request.run_id,
                call_id=request.call_id,
                action=request.action,
                status=ActionStatus.CONFIRMATION_REQUIRED,
                message="Explicit confirmation is required for this action.",
                confirmation_id=security.confirmation_id,
            )
            self._publish("waiting_confirmation", request, observation.status)
            return observation
        if security.status is SecurityDecisionStatus.BLOCKED:
            observation = ActionObservation(
                run_id=request.run_id,
                call_id=request.call_id,
                action=request.action,
                status=ActionStatus.BLOCKED,
                message="Action blocked by the security gate.",
            )
            self._publish("blocked", request, observation.status)
            return observation

        if request.idempotency_key:
            idempotency_key = (request.action, request.idempotency_key)
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                observation = existing.model_copy(update={"deduplicated": True})
                self._completed[call_key] = observation
                self._publish("deduplicated", request, observation.status)
                return observation

        started_at = self._clock()
        self._publish("started", request, None)
        try:
            receipt = self._router.execute(
                request.action,
                normalized_request.arguments,
                correlation_id=request.correlation_id or request.run_id,
                metadata={
                    **request.metadata,
                    "agent_run_id": request.run_id,
                    "tool_call_id": request.call_id,
                },
            )
        except ActionExecutionError:
            observation = self._failure(
                request,
                "ActionExecutionError",
                latency_ms=(self._clock() - started_at) * 1000.0,
            )
        else:
            status = (
                ActionStatus.SUCCESS
                if receipt.status is ExecutionStatus.SUCCESS
                else ActionStatus.FAILED
            )
            observation = ActionObservation(
                run_id=request.run_id,
                call_id=request.call_id,
                action=request.action,
                status=status,
                message=receipt.message,
                data=receipt.data,
                artifacts=receipt.artifacts,
                latency_ms=max((self._clock() - started_at) * 1000.0, 0.0),
            )
            self._publish(
                "succeeded" if status is ActionStatus.SUCCESS else "failed",
                request,
                status,
                observation.latency_ms,
            )
        self._completed[call_key] = observation
        if request.idempotency_key and observation.status is ActionStatus.SUCCESS:
            self._idempotency[(request.action, request.idempotency_key)] = observation
        return observation

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.payload_model.model_json_schema(),
                },
            }
            for definition in self._router.registry.list_actions()
        ]

    def _failure(
        self,
        request: ActionRequest,
        error_type: str,
        *,
        latency_ms: float = 0.0,
    ) -> ActionObservation:
        observation = ActionObservation(
            run_id=request.run_id,
            call_id=request.call_id,
            action=request.action,
            status=ActionStatus.FAILED,
            message="Action execution failed.",
            error_type=error_type,
            latency_ms=max(latency_ms, 0.0),
        )
        self._publish("failed", request, observation.status, observation.latency_ms)
        return observation

    def _publish(
        self,
        suffix: str,
        request: ActionRequest,
        status: ActionStatus | None,
        latency_ms: float | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        payload: dict[str, Any] = {
            "run_id": request.run_id,
            "call_id": request.call_id,
            "correlation_id": request.correlation_id,
            "action": request.action,
            "status": status.value if status is not None else "active",
        }
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 3)
        self._event_sink(f"action_executor.{suffix}", payload)
