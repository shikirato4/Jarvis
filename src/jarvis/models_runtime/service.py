from __future__ import annotations

import logging
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, ModelProviderError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .base import ModelRequest, ModelResponse, ProviderHealth
from .catalog import ModelCatalog
from .registry import ProviderRegistry
from .router import ModelRouter


class ModelService:
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
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._provider_registry = provider_registry
        self._catalog = catalog
        self._router = router
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.models")
        self._resilience = resilience_controller

    def list_models(self) -> list[dict[str, Any]]:
        return [profile.model_dump(mode="json") for profile in self._catalog.list_profiles()]

    def health(self) -> list[ProviderHealth]:
        return [provider.health_check() for provider in self._provider_registry.list_providers()]

    def infer(self, request: ModelRequest) -> ModelResponse:
        if request.stream and not self._mode_manager.current_policy().streaming_allowed:
            raise CapabilityUnavailableError(
                "streaming inference is not allowed in the current mode",
                details={"mode": self._mode_manager.current_mode().value},
            )
        requested_logical_model = request.logical_model or "auto"
        strict_provider = self._strict_provider_for_request(request)
        candidates = self._router.route(request, preferred_provider_order=self._preferred_provider_order_for_request(request))
        original_candidates = list(candidates)
        if strict_provider is not None:
            blocked_candidates = [profile for profile in candidates if profile.provider.casefold() != strict_provider.casefold()]
            if blocked_candidates:
                self._logger.warning(
                    "model_provider_blocked",
                    extra={
                        "correlation_id": request.correlation_id,
                        "logical_model": requested_logical_model,
                        "task_type": request.task_type,
                        "required_provider": strict_provider,
                        "blocked_providers": [profile.provider for profile in blocked_candidates],
                    },
                )
            candidates = [profile for profile in candidates if profile.provider.casefold() == strict_provider.casefold()]
        self._logger.info(
            "model_route_selected",
            extra={
                "correlation_id": request.correlation_id,
                "logical_model": requested_logical_model,
                "task_type": request.task_type,
                "strict_provider": strict_provider,
                "candidate_providers": [profile.provider for profile in candidates],
                "candidate_models": [profile.model_name for profile in candidates],
                "original_candidate_providers": [profile.provider for profile in original_candidates],
            },
        )
        if not candidates:
            if strict_provider is not None:
                message = f"general conversation requires provider '{strict_provider}', but no eligible model route is available"
                self._logger.error(
                    "model_route_unavailable",
                    extra={
                        "correlation_id": request.correlation_id,
                        "logical_model": requested_logical_model,
                        "task_type": request.task_type,
                        "required_provider": strict_provider,
                    },
                )
                raise ModelProviderError(message)
            raise CapabilityUnavailableError("no model candidates available", details={"task_type": request.task_type})

        last_error: Exception | None = None
        for index, profile in enumerate(candidates):
            provider = self._provider_registry.get(profile.provider)
            if provider is None:
                last_error = ModelProviderError(f"required provider '{profile.provider}' is not registered")
                self._logger.error(
                    "model_provider_missing",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": profile.provider,
                        "logical_model": profile.logical_name,
                        "task_type": request.task_type,
                    },
                )
                continue
            try:
                self._logger.info(
                    "model_provider_attempt",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": profile.provider,
                        "logical_model": profile.logical_name,
                        "model_name": profile.model_name,
                        "task_type": request.task_type,
                        "fallback_attempt": index > 0,
                        "base_url": self._provider_base_url(profile.provider),
                        "endpoint": self._provider_endpoint(profile.provider),
                    },
                )
                if self._resilience is not None:
                    response, _ = self._resilience.execute(
                        service_name="models_runtime",
                        dependency_name=profile.provider,
                        operation_name="infer",
                        timeout_ms=int(((request.timeout_seconds if request.timeout_seconds is not None else profile.timeout_seconds) or 0) * 1000) or None,
                        func=lambda provider=provider, profile=profile: provider.infer(
                            request,
                            model_name=profile.model_name,
                            temperature=request.temperature if request.temperature is not None else profile.temperature,
                            timeout_seconds=request.timeout_seconds if request.timeout_seconds is not None else profile.timeout_seconds,
                        ),
                    )
                else:
                    response = provider.infer(
                        request,
                        model_name=profile.model_name,
                        temperature=request.temperature if request.temperature is not None else profile.temperature,
                        timeout_seconds=request.timeout_seconds if request.timeout_seconds is not None else profile.timeout_seconds,
                    )
                response.logical_model = profile.logical_name
                response.fallback_used = index > 0
                response.metadata.update(
                    {
                        "purpose": profile.purpose,
                        "task_type": request.task_type,
                        "fallback_index": index,
                        "correlation_id": request.correlation_id,
                        "base_url": self._provider_base_url(profile.provider),
                    }
                )
                self._logger.info(
                    "model_provider_success",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": response.provider_name,
                        "logical_model": response.logical_model,
                        "model_name": response.model_name,
                        "task_type": request.task_type,
                        "fallback_used": response.fallback_used,
                        "base_url": self._provider_base_url(profile.provider),
                        "endpoint": self._provider_endpoint(profile.provider),
                    },
                )
                self._event_bus.publish(
                    "model.executed",
                    {
                        "correlation_id": request.correlation_id,
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
            except Exception as exc:
                last_error = exc
                self._logger.exception(
                    "model_provider_failed",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": profile.provider,
                        "logical_model": profile.logical_name,
                        "model_name": profile.model_name,
                        "task_type": request.task_type,
                        "fallback_attempt": index > 0,
                        "base_url": self._provider_base_url(profile.provider),
                        "endpoint": self._provider_endpoint(profile.provider),
                    },
                )
                self._event_bus.publish(
                    "model.failed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": profile.provider,
                        "logical_model": profile.logical_name,
                        "model_name": profile.model_name,
                        "task_type": request.task_type,
                        "error": str(exc),
                    },
                )
        if last_error is None:
            raise CapabilityUnavailableError("no registered providers available", details={"task_type": request.task_type})
        if strict_provider is not None:
            raise ModelProviderError(f"general conversation requires provider '{strict_provider}' and the request failed: {last_error}")
        raise ModelProviderError(str(last_error))

    def _strict_provider_for_request(self, request: ModelRequest) -> str | None:
        if request.logical_model == "general_assistant" or request.task_type == "assistant":
            if self._settings.general_chat_model_fallback_order:
                return None
            return self._settings.general_chat_model_provider
        return None

    def _preferred_provider_order_for_request(self, request: ModelRequest) -> tuple[str, ...]:
        if request.logical_model == "general_assistant" or request.task_type == "assistant":
            if self._settings.general_chat_model_fallback_order:
                return (self._settings.general_chat_model_provider,)
        return self._settings.model_provider_fallback_order

    def _provider_base_url(self, provider_name: str) -> str | None:
        if provider_name == "gpt_oss":
            return self._settings.gpt_oss_base_url
        if provider_name == "ollama":
            return self._settings.ollama_base_url
        return None

    def _provider_endpoint(self, provider_name: str) -> str | None:
        if provider_name == "gpt_oss":
            return self._settings.gpt_oss_chat_endpoint
        if provider_name == "ollama":
            return self._settings.ollama_chat_endpoint
        return None
