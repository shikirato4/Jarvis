from __future__ import annotations

import logging
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, JarvisError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .backends import STTProviderRegistry
from .base import STTProvider, TranscriptionRequest, TranscriptionResult


class STTService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        registry: STTProviderRegistry,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
        resilience_controller=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._registry = registry
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.voice.stt")
        self._resilience = resilience_controller

    def health(self) -> list[dict[str, Any]]:
        return [provider.health_check() for provider in self._registry.list_providers()]

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        candidates = self._candidate_providers(request)
        if not candidates:
            raise CapabilityUnavailableError("no stt providers available")
        last_error: Exception | None = None
        for index, provider in enumerate(candidates):
            try:
                if self._resilience is not None:
                    result, _ = self._resilience.execute(
                        service_name="voice_runtime",
                        dependency_name=provider.provider_name,
                        operation_name="stt.transcribe",
                        timeout_ms=int(getattr(request, "timeout_seconds", 0) * 1000) if getattr(request, "timeout_seconds", None) is not None else None,
                        func=lambda provider=provider: provider.transcribe(request),
                    )
                else:
                    result = provider.transcribe(request)
                result.fallback_used = index > 0
                self._event_bus.publish(
                    "voice.stt.executed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "provider_kind": provider.provider_kind,
                        "latency_ms": result.latency_ms,
                        "fallback_used": result.fallback_used,
                        "text_length": len(result.text),
                    },
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.exception("voice_stt_failed", extra={"provider": provider.provider_name})
                self._event_bus.publish(
                    "voice.stt.failed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "provider_kind": provider.provider_kind,
                        "error": str(exc),
                    },
                )
        assert last_error is not None
        raise JarvisError(str(last_error), component="voice_runtime", code="stt_failed", recoverable=True)

    def _candidate_providers(self, request: TranscriptionRequest) -> list[STTProvider]:
        request_preferred = str(request.metadata.get("preferred_stt_provider", "")).strip() or None
        preferred = tuple(
            name
            for name in (
                request_preferred,
                self._settings.voice_stt_provider_default,
                *self._settings.voice_stt_provider_fallback_order,
            )
            if name
        )
        seen: set[str] = set()
        providers: list[STTProvider] = []
        for name in preferred:
            if not name or name in seen:
                continue
            provider = self._registry.get(name)
            if provider is not None:
                providers.append(provider)
                seen.add(name)
        if providers:
            return providers
        return self._registry.list_providers()
