from __future__ import annotations

import logging
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import (
    CapabilityUnavailableError,
    ConfigurationError,
    ModelProviderError,
)
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .base import ModelRequest, ModelResponse, ProviderHealth
from .catalog import ModelCatalog
from .coordinator import HybridProviderUnavailableError, ProviderCoordinator
from .hybrid import LLMMode, ProviderSelection, RequestExecutionContext
from .registry import ProviderRegistry
from .router import ModelRouter


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
            response = self._coordinator.infer(
                request,
                execution_context=execution_context,
            )
        except HybridProviderUnavailableError as exc:
            self._log_route("model_request_failed", request_id=request_id)
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
        request = request.model_copy(update={"stream": True})
        request_id = (
            execution_context.request_id
            if execution_context is not None
            else request.correlation_id
        )
        try:
            for chunk in self._coordinator.stream(
                request,
                execution_context=execution_context,
                cancel_check=cancel_check,
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
                            "route_generation": "coordinator",
                            "incomplete": bool(chunk.error),
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
                "route_generation": "coordinator",
            },
        )
