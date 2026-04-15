from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, ConfigurationError, EmbeddingProviderError, EmbeddingRoutingError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .base import EmbeddingProvider, EmbeddingProviderHealth, EmbeddingProfile, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class EmbeddingProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, EmbeddingProvider] = {}

    def register(self, provider: EmbeddingProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"embedding provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> EmbeddingProvider | None:
        return self._providers.get(provider_name)

    def list_providers(self) -> list[EmbeddingProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class EmbeddingRouter:
    def __init__(self, profiles: list[EmbeddingProfile], mode_manager: ModeManager) -> None:
        self._profiles = {profile.logical_name: profile for profile in profiles}
        self._mode_manager = mode_manager

    def route(self, request: EmbeddingRequest, *, preferred_provider_order: tuple[str, ...]) -> list[EmbeddingProfile]:
        allowed_provider_kinds = set(self._mode_manager.current_policy().allowed_provider_kinds)
        ordered = self._select_profiles(request)
        if allowed_provider_kinds:
            ordered = [profile for profile in ordered if profile.provider_kind in allowed_provider_kinds]
        if preferred_provider_order:
            priority = {name: index for index, name in enumerate(preferred_provider_order)}
            ordered.sort(key=lambda item: (priority.get(item.provider, len(priority)), item.priority, item.logical_name))
        return ordered

    def list_profiles(self) -> list[EmbeddingProfile]:
        return sorted(self._profiles.values(), key=lambda item: (item.priority, item.logical_name))

    def _select_profiles(self, request: EmbeddingRequest) -> list[EmbeddingProfile]:
        if request.logical_model:
            primary = self._profiles.get(request.logical_model)
            if primary is None:
                raise EmbeddingRoutingError(f"embedding profile '{request.logical_model}' is not configured")
            ordered = [primary]
            ordered.extend(self._profiles[name] for name in primary.fallbacks if name in self._profiles)
            return ordered
        matches = [profile for profile in self.list_profiles() if not profile.task_types or request.task_type in profile.task_types]
        return matches or self.list_profiles()


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        provider_registry: EmbeddingProviderRegistry,
        router: EmbeddingRouter,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
        resilience_controller=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._provider_registry = provider_registry
        self._router = router
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.embeddings")
        self._resilience = resilience_controller

    def list_models(self) -> list[dict[str, Any]]:
        return [profile.model_dump(mode="json") for profile in self._router.list_profiles()]

    def health(self) -> list[EmbeddingProviderHealth]:
        return [provider.health_check() for provider in self._provider_registry.list_providers()]

    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        candidates = self._router.route(request, preferred_provider_order=self._settings.embedding_provider_fallback_order)
        if not candidates:
            raise CapabilityUnavailableError("no embedding candidates available", details={"task_type": request.task_type})

        last_error: Exception | None = None
        for index, profile in enumerate(candidates):
            provider = self._provider_registry.get(profile.provider)
            if provider is None:
                continue
            started_at = time.perf_counter()
            try:
                if self._resilience is not None:
                    response, _ = self._resilience.execute(
                        service_name="semantic_memory",
                        dependency_name=profile.provider,
                        operation_name="embed",
                        timeout_ms=int((request.timeout_seconds or 0) * 1000) or None,
                        func=lambda provider=provider, profile=profile: provider.embed(
                            request,
                            model_name=profile.model_name,
                            timeout_seconds=request.timeout_seconds,
                        ),
                    )
                else:
                    response = provider.embed(
                        request,
                        model_name=profile.model_name,
                        timeout_seconds=request.timeout_seconds,
                    )
                response.logical_model = profile.logical_name
                response.fallback_used = index > 0
                response.metadata.update(
                    {
                        "purpose": profile.purpose,
                        "task_type": request.task_type,
                        "fallback_index": index,
                        "correlation_id": request.correlation_id,
                    }
                )
                self._event_bus.publish(
                    "embedding.executed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": response.provider_name,
                        "provider_kind": response.provider_kind,
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
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                self._logger.exception(
                    "embedding_provider_failed",
                    extra={
                        "provider": profile.provider,
                        "logical_model": profile.logical_name,
                        "task_type": request.task_type,
                        "latency_ms": elapsed_ms,
                    },
                )
                self._event_bus.publish(
                    "embedding.failed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": profile.provider,
                        "provider_kind": profile.provider_kind,
                        "logical_model": profile.logical_name,
                        "model_name": profile.model_name,
                        "task_type": request.task_type,
                        "latency_ms": elapsed_ms,
                        "error": str(exc),
                    },
                )
        if last_error is None:
            raise CapabilityUnavailableError("no registered embedding providers available")
        raise EmbeddingProviderError(str(last_error))


class OllamaEmbeddingProvider:
    provider_name = "ollama_embeddings"
    provider_kind = "local"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self._base_url)

    def health_check(self) -> EmbeddingProviderHealth:
        try:
            response = self._client.get(
                self._settings.ollama_tags_endpoint,
                timeout=self._settings.ollama_healthcheck_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return EmbeddingProviderHealth(
                provider_name=self.provider_name,
                healthy=True,
                details={"models": len(payload.get("models", []))},
            )
        except Exception as exc:
            return EmbeddingProviderHealth(
                provider_name=self.provider_name,
                healthy=False,
                details={"error": str(exc)},
            )

    def embed(
        self,
        request: EmbeddingRequest,
        *,
        model_name: str,
        timeout_seconds: float | None,
    ) -> EmbeddingResponse:
        attempts = max(self._settings.ollama_max_retries + 1, 1)
        timeout = timeout_seconds or self._settings.ollama_timeout_seconds
        payload = {"model": model_name, "input": list(request.texts)}
        last_error: Exception | None = None
        started_at = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(
                    self._settings.ollama_embeddings_endpoint,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                vectors = [
                    EmbeddingVector(
                        index=index,
                        text=request.texts[index],
                        values=[float(value) for value in values],
                        dimensions=len(values),
                    )
                    for index, values in enumerate(body.get("embeddings", []))
                ]
                return EmbeddingResponse(
                    provider_name=self.provider_name,
                    provider_kind=self.provider_kind,
                    logical_model=request.logical_model or model_name,
                    model_name=body.get("model", model_name),
                    vectors=vectors,
                    latency_ms=elapsed_ms,
                    metadata={"attempt": attempt, "task_type": request.task_type},
                )
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(self._settings.ollama_retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error


def build_default_embedding_profiles(settings: Settings) -> list[EmbeddingProfile]:
    return [
        EmbeddingProfile(
            logical_name="general_embedding",
            provider=settings.embedding_provider_default,
            provider_kind="local",
            model_name=settings.embedding_model_default,
            purpose="General semantic retrieval and document indexing.",
            dimensions=None,
            task_types=("retrieval", "ingestion"),
            priority=10,
            fallbacks=(),
        )
    ]
