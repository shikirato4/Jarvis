from __future__ import annotations

import logging
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import (
    CapabilityUnavailableError,
    ConfigurationError,
    ContextLimitError,
    ModelProviderError,
)
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .base import ModelRequest, ModelResponse, ProviderHealth
from .catalog import ModelCatalog
from .coordinator import HybridProviderUnavailableError, ProviderCoordinator
from .errors import classify_provider_failure
from .hybrid import LLMMode, ProviderSelection, RequestExecutionContext
from .output_completion import OutputCompletionPolicy, estimate_request_tokens
from .registry import ProviderRegistry
from .router import ModelRouter
from .roles import LLMTaskRouter, ModelRoleRegistry


class ModelService:
    """Provider-neutral model facade backed exclusively by the hybrid coordinator."""

    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        provider_registry: ProviderRegistry,
        catalog: ModelCatalog,
        router: ModelRouter,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
        resilience_controller=None,
        coordinator: ProviderCoordinator | None = None,
        model_roles: ModelRoleRegistry | None = None,
    ) -> None:
        if coordinator is None:
            raise ConfigurationError(
                "ModelService requires a ProviderCoordinator for AUTO, NVIDIA API, or LOCAL."
            )
        self._settings = settings
        self._mode_manager = mode_manager
        self._provider_registry = provider_registry
        self._catalog = catalog
        self._router = router
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.models")
        self._resilience = resilience_controller
        self._coordinator = coordinator
        self._model_roles = model_roles
        self._output_completion = OutputCompletionPolicy(settings)

    def list_model_roles(self) -> dict[str, dict[str, Any]]:
        return self._model_roles.describe() if self._model_roles is not None else {}

    def discover_models(self) -> list[dict[str, Any]]:
        """Explicit discovery call; no model-list HTTP requests run per inference."""
        discovered: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for provider in self._provider_registry.list_providers():
            list_models = getattr(provider, "discover_models", None)
            if not callable(list_models):
                continue
            for item in list_models():
                identity = (str(item.get("provider", provider.provider_name)), str(item.get("identifier", "")))
                if identity not in seen:
                    seen.add(identity)
                    discovered.append(item)
        return discovered

    def switch_model(self, role: str, model: str) -> str:
        """Hot-switch a role only when its configured backend advertises the model."""
        if self._model_roles is not None and self._model_roles.is_configured(role):
            return self._model_roles.switch_model(role, model)
        if role == "primary":
            return self._coordinator.switch_local_model(model)
        raise ConfigurationError(f"No configurable LLM model role is registered: {role}")

    def chat(self, request: ModelRequest, *, execution_context: RequestExecutionContext | None = None) -> ModelResponse:
        """Provider-neutral chat API; existing callers may keep using ``infer``."""
        return self.infer(request, execution_context=execution_context)

    def generate(
        self,
        prompt: str,
        *,
        task_type: str = "assistant",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ModelResponse:
        return self.infer(ModelRequest(
            task_type=task_type,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ))

    def for_task(self, task_type: str) -> "TaskModelClient":
        return TaskModelClient(self, task_type)

    def _use_role_router(self, request: ModelRequest) -> bool:
        if self._model_roles is None:
            return False
        role = LLMTaskRouter.role_for(request)
        if request.metadata.get("role_context_exceeded"):
            return False
        if not self._model_roles.is_configured(role) and not (
            role == "vision" and self._model_roles.roles
        ):
            return False
        allowed = {
            item.casefold()
            for item in self._mode_manager.current_policy().allowed_provider_kinds
        }
        return "local" in allowed and role != "primary"

    def _prepare_model_request(self, request: ModelRequest) -> ModelRequest:
        if self._model_roles is None:
            return request
        role = LLMTaskRouter.role_for(request)
        context_limit = self._model_roles.context_limit(role)
        if context_limit is None:
            return request
        estimated_input = estimate_request_tokens(request)
        safety = max(int(getattr(self._settings, "conversation_context_safety_margin_tokens", 256)), 128)
        required_output = request.max_tokens if request.max_tokens is not None else 32
        if estimated_input + safety + required_output > context_limit:
            details = {
                "role": role,
                "context_limit_tokens": context_limit,
                "estimated_input_tokens": estimated_input,
                "requested_output_tokens": required_output,
                "safety_margin_tokens": safety,
            }
            if role != "primary":
                metadata = dict(request.metadata)
                metadata["role_context_exceeded"] = details
                fallback_context = self._model_roles.context_limit("primary") or int(
                    getattr(self._settings, "local_runtime_context_size", context_limit)
                )
                if estimated_input + safety + required_output > fallback_context:
                    raise ContextLimitError(
                        "Request exceeds the available primary model context window",
                        details={**details, "fallback_context_limit_tokens": fallback_context},
                    )
                metadata["output_context_tokens"] = fallback_context
                return request.model_copy(update={"metadata": metadata})
            raise ContextLimitError(
                f"Request exceeds the configured context window for role {role!r}",
                details=details,
            )
        metadata = dict(request.metadata)
        metadata["output_context_tokens"] = context_limit
        return request.model_copy(update={"metadata": metadata})

    def list_models(self) -> list[dict[str, Any]]:
        return [profile.model_dump(mode="json") for profile in self._catalog.list_profiles()]

    def health(self) -> list[ProviderHealth]:
        return [
            provider.health_check()
            for provider in self._provider_registry.list_providers()
        ]

    def provider_selection(self) -> ProviderSelection:
        return self._coordinator.selection_snapshot()

    def provider_statuses(self):
        return self._coordinator.provider_statuses()

    def set_llm_mode(self, mode: LLMMode | str) -> LLMMode:
        return self._coordinator.set_mode(mode)

    def infer(
        self,
        request: ModelRequest,
        *,
        execution_context: RequestExecutionContext | None = None,
    ) -> ModelResponse:
        request = self._output_completion.prepare_request(
            self._prepare_model_request(request)
        )
        if request.stream and not self._mode_manager.current_policy().streaming_allowed:
            raise CapabilityUnavailableError(
                "streaming inference is not allowed in the current mode",
                details={"mode": self._mode_manager.current_mode().value},
            )
        request_id = (
            execution_context.request_id
            if execution_context is not None
            else request.correlation_id
        )
        try:
            role_routed = self._use_role_router(request)

            def infer_once(item: ModelRequest) -> ModelResponse:
                if role_routed:
                    return self._model_roles.infer(
                        item,
                        role=LLMTaskRouter.role_for(item),
                        primary_infer=lambda primary: self._coordinator.infer(
                            primary, execution_context=execution_context
                        ),
                    )
                return self._coordinator.infer(
                    item, execution_context=execution_context
                )
            response = infer_once(request)
            response = self._output_completion.complete_response(
                request,
                response,
                infer_next=infer_once,
                event_sink=self._event_bus.publish,
            )
        except HybridProviderUnavailableError as exc:
            self._log_route("model_request_failed", request_id=request_id)
            failure = classify_provider_failure(exc.__cause__ or exc)
            self._logger.error(
                "model_provider_failure",
                extra={
                    "request_id": request_id,
                    "provider": failure.provider,
                    "category": failure.category.value,
                    "status_code": failure.status_code,
                    "retryable": failure.retryable,
                    "fallback_eligible": failure.fallback_eligible,
                    "retry_after_seconds": failure.retry_after_seconds,
                    "safe_message": failure.detail,
                },
            )
            raise ModelProviderError(str(exc)) from exc
        self._log_route(
            "model_request_completed",
            request_id=request_id,
            response=response,
        )
        self._event_bus.publish(
            "model.executed",
            {
                "correlation_id": request_id,
                "provider": response.provider_name,
                "provider_kind": response.provider_kind.value,
                "logical_model": response.logical_model,
                "model_name": response.model_name,
                "latency_ms": response.latency_ms,
                "fallback_used": response.fallback_used,
                "task_type": request.task_type,
            },
        )
        return response

    def stream(
        self,
        request: ModelRequest,
        *,
        cancel_check=None,
        execution_context: RequestExecutionContext | None = None,
    ):
        request = self._output_completion.prepare_request(
            self._prepare_model_request(request.model_copy(update={"stream": True}))
        )
        request_id = (
            execution_context.request_id
            if execution_context is not None
            else request.correlation_id
        )
        try:
            role_routed = self._use_role_router(request)

            def stream_once(item: ModelRequest):
                if role_routed:
                    return self._model_roles.stream(
                        item,
                        role=LLMTaskRouter.role_for(item),
                        primary_stream=lambda primary: self._coordinator.stream(
                            primary,
                            execution_context=execution_context,
                            cancel_check=cancel_check,
                        ),
                        cancel_check=cancel_check,
                    )
                return self._coordinator.stream(
                    item,
                    execution_context=execution_context,
                    cancel_check=cancel_check,
                )
            for chunk in self._output_completion.stream_completed(
                request,
                stream_once=stream_once,
                event_sink=self._event_bus.publish,
            ):
                if chunk.done:
                    selection = self._coordinator.selection_snapshot()
                    self._logger.info(
                        "model_stream_finished",
                        extra={
                            "request_id": request_id,
                            "selected_mode": selection.selected_mode.value,
                            "request_provider": selection.request_provider,
                            "active_provider": selection.active_provider,
                            "model_name": selection.active_model,
                            "fallback_reason": selection.fallback_reason,
                            "route_generation": chunk.metadata.get("route", "coordinator"),
                            "model_role": chunk.metadata.get("model_role"),
                            "incomplete": bool(
                                chunk.error or chunk.metadata.get("incomplete")
                            ),
                            "finish_reason": chunk.metadata.get("finish_reason"),
                            "stream_closed_normally": chunk.metadata.get(
                                "stream_closed_normally"
                            ),
                        },
                    )
                yield chunk
        except HybridProviderUnavailableError as exc:
            self._log_route("model_stream_failed", request_id=request_id)
            raise ModelProviderError(str(exc)) from exc

    def _log_route(
        self,
        event: str,
        *,
        request_id: str | None,
        response: ModelResponse | None = None,
    ) -> None:
        selection = self._coordinator.selection_snapshot()
        self._logger.info(
            event,
            extra={
                "request_id": request_id,
                "selected_mode": selection.selected_mode.value,
                "request_provider": selection.request_provider,
                "active_provider": (
                    response.provider_name if response is not None else selection.active_provider
                ),
                "model_name": (
                    response.model_name if response is not None else selection.active_model
                ),
                "fallback_reason": selection.fallback_reason,
                "route_generation": (
                    response.metadata.get("route", "coordinator")
                    if response is not None
                    else "coordinator"
                ),
                "model_role": response.metadata.get("model_role") if response else None,
            },
        )


class TaskModelClient:
    """Small task-bound view over ModelService for provider-neutral callers."""

    def __init__(self, service: ModelService, task_type: str) -> None:
        self._service = service
        self._task_type = task_type

    def chat(self, request: ModelRequest, *, execution_context: RequestExecutionContext | None = None) -> ModelResponse:
        return self._service.chat(
            request.model_copy(update={"task_type": self._task_type}),
            execution_context=execution_context,
        )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return self._service.generate(prompt, task_type=self._task_type, **kwargs)

    def stream(self, request: ModelRequest, **kwargs):
        return self._service.stream(
            request.model_copy(update={"task_type": self._task_type}), **kwargs
        )
