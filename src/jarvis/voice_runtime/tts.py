from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, JarvisError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager

from .backends import TTSProviderRegistry
from .base import SynthesisRequest, SynthesisResult, TTSProvider


class TTSService:
    def __init__(
        self,
        settings: Settings,
        mode_manager: ModeManager,
        registry: TTSProviderRegistry,
        event_bus: EventBus,
        logger: logging.Logger | None = None,
        resilience_controller=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._registry = registry
        self._event_bus = event_bus
        self._logger = logger or logging.getLogger("jarvis.voice.tts")
        self._resilience = resilience_controller
        self._provider_backoff_until: dict[str, float] = {}

    def health(self) -> list[dict[str, Any]]:
        return [provider.health_check() for provider in self._registry.list_providers()]

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        candidates = self._candidate_providers(request)
        if not candidates:
            raise CapabilityUnavailableError("no tts providers available")
        last_error: Exception | None = None
        failure_details: list[dict[str, str]] = []
        metadata = request.metadata or {}
        selected_names = [provider.provider_name for provider in candidates]
        self._logger.info(
            "tts_provider_selected",
            extra={
                "correlation_id": request.correlation_id,
                "preferred_provider": request.provider_name or self._settings.voice_tts_provider_default,
                "candidate_providers": selected_names,
                "profile_name": request.profile_name,
                "voice_name": request.voice_name,
                "rate": request.rate,
                "speaker_name": metadata.get("speaker_name"),
                "speaker_wav": metadata.get("speaker_wav"),
                "speaker_wav_effective": metadata.get("speaker_wav_effective") or metadata.get("speaker_wav"),
                "speaking_rate": metadata.get("speaking_rate"),
            },
        )
        for index, provider in enumerate(candidates):
            try:
                self._logger.info(
                    "voice_tts_attempt",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "attempt_index": index,
                        "is_fallback": index > 0,
                        "profile_name": request.profile_name,
                        "speaker_name": metadata.get("speaker_name"),
                        "speaker_wav": metadata.get("speaker_wav"),
                        "speaking_rate": metadata.get("speaking_rate"),
                        "rate": request.rate,
                    },
                )
                if self._resilience is not None:
                    result, _ = self._resilience.execute(
                        service_name="voice_runtime",
                        dependency_name=provider.provider_name,
                        operation_name="tts.synthesize",
                        timeout_ms=self._resolve_timeout_ms(request),
                        func=lambda provider=provider: provider.synthesize(request),
                    )
                else:
                    result = provider.synthesize(request)
                result.fallback_used = index > 0
                result.metadata["requested_provider"] = self._coalesce_metadata_value(
                    result.metadata.get("requested_provider"),
                    request.provider_name or self._settings.voice_tts_provider_default,
                )
                result.metadata["profile_name"] = self._coalesce_metadata_value(result.metadata.get("profile_name"), request.profile_name)
                result.metadata["voice_name"] = self._coalesce_metadata_value(result.metadata.get("voice_name"), request.voice_name)
                result.metadata["rate"] = self._coalesce_metadata_value(result.metadata.get("rate"), request.rate)
                result.metadata["speaker_name"] = self._coalesce_metadata_value(result.metadata.get("speaker_name"), metadata.get("speaker_name"))
                result.metadata["speaker_wav"] = self._coalesce_metadata_value(result.metadata.get("speaker_wav"), metadata.get("speaker_wav"))
                result.metadata["speaking_rate"] = self._coalesce_metadata_value(
                    result.metadata.get("speaking_rate"),
                    metadata.get("speaking_rate"),
                )
                result.metadata["style"] = self._coalesce_metadata_value(result.metadata.get("style"), metadata.get("style"))
                result.metadata["attempted_providers"] = selected_names
                result.metadata["fallback_reason"] = failure_details[-1]["reason"] if failure_details else None
                self._logger.info(
                    "voice_tts_success",
                    extra={
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "provider_kind": provider.provider_kind,
                        "fallback_used": result.fallback_used,
                        "attempted_providers": selected_names,
                        "speaker_name_effective": result.metadata.get("speaker_name") or result.metadata.get("speaker"),
                        "speaker_wav_effective": result.metadata.get("speaker_wav"),
                        "speaking_rate_effective": result.metadata.get("speaking_rate"),
                        "rate_effective": result.metadata.get("rate"),
                    },
                )
                if result.fallback_used and failure_details:
                    self._logger.warning(
                        "voice_provider_fallback",
                        extra={
                            "correlation_id": request.correlation_id,
                            "from_provider": failure_details[-1]["provider"],
                            "to_provider": provider.provider_name,
                            "reason": failure_details[-1]["reason"],
                        },
                    )
                self._event_bus.publish(
                    "voice.tts.executed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "provider_kind": provider.provider_kind,
                        "latency_ms": result.latency_ms,
                        "fallback_used": result.fallback_used,
                        "text_length": len(request.text),
                    },
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._mark_provider_backoff(provider.provider_name, exc)
                failure_details.append({"provider": provider.provider_name, "reason": str(exc)})
                self._logger.exception("voice_tts_failed", extra={"provider": provider.provider_name})
                self._event_bus.publish(
                    "voice.tts.failed",
                    {
                        "correlation_id": request.correlation_id,
                        "provider": provider.provider_name,
                        "provider_kind": provider.provider_kind,
                        "error": str(exc),
                    },
                )
        assert last_error is not None
        raise JarvisError(str(last_error), component="voice_runtime", code="tts_failed", recoverable=True)

    @staticmethod
    def _coalesce_metadata_value(current: object, fallback: object) -> object:
        if current not in (None, ""):
            return current
        return fallback

    def _candidate_providers(self, request: SynthesisRequest) -> list[TTSProvider]:
        preferred = (
            request.provider_name or self._settings.voice_tts_provider_default,
            *request.fallback_provider_names,
            *self._settings.voice_tts_provider_fallback_order,
        )
        seen: set[str] = set()
        providers: list[TTSProvider] = []
        now = perf_counter()
        for name in preferred:
            if not name or name in seen:
                continue
            if self._provider_backoff_until.get(name, 0.0) > now:
                continue
            provider = self._registry.get(name)
            if provider is not None:
                providers.append(provider)
                seen.add(name)
        if providers:
            return providers
        return [provider for provider in self._registry.list_providers() if self._provider_backoff_until.get(provider.provider_name, 0.0) <= now]

    def _resolve_timeout_ms(self, request: SynthesisRequest) -> int | None:
        timeout_seconds = getattr(request, "timeout_seconds", None)
        if timeout_seconds is not None:
            return int(timeout_seconds * 1000)
        return self._settings.voice_watchdog_timeout_ms

    def _mark_provider_backoff(self, provider_name: str, exc: Exception) -> None:
        if not self._should_back_off(exc):
            return
        self._provider_backoff_until[provider_name] = perf_counter() + 30.0

    @staticmethod
    def _should_back_off(exc: Exception) -> bool:
        text = str(exc).casefold()
        return any(
            token in text
            for token in (
                "package not installed",
                "module not found",
                "no module named",
                "unavailable",
                "missing speaker_wav",
            )
        )
