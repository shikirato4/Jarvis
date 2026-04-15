from __future__ import annotations

import logging
from typing import Any

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError, ConfigurationError, JarvisError

from .base import OCRProvider, OCRRequest, OCRResult, ScreenCaptureResult


class OCRProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, OCRProvider] = {}

    def register(self, provider: OCRProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"ocr provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> OCRProvider | None:
        return self._providers.get(provider_name)

    def list_providers(self) -> list[OCRProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class OCRService:
    def __init__(self, settings: Settings, registry: OCRProviderRegistry, logger: logging.Logger | None = None, resilience_controller=None) -> None:
        self._settings = settings
        self._registry = registry
        self._logger = logger or logging.getLogger("jarvis.vision.ocr")
        self._resilience = resilience_controller

    def health(self) -> list[dict[str, Any]]:
        return [provider.health_check() for provider in self._registry.list_providers()]

    def extract_text(self, request: OCRRequest, *, capture: ScreenCaptureResult | None = None) -> OCRResult:
        candidates = self._candidate_providers(request.provider_name)
        if not candidates:
            raise CapabilityUnavailableError("no ocr providers available", component="vision_runtime")
        last_error: Exception | None = None
        for index, provider in enumerate(candidates):
            try:
                if self._resilience is not None:
                    result, _ = self._resilience.execute(
                        service_name="vision_runtime",
                        dependency_name=provider.provider_name,
                        operation_name="ocr.extract_text",
                        func=lambda provider=provider: provider.extract_text(request, capture=capture),
                    )
                else:
                    result = provider.extract_text(request, capture=capture)
                result.fallback_used = index > 0
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.exception("vision_ocr_provider_failed", extra={"provider": provider.provider_name})
        assert last_error is not None
        raise JarvisError(str(last_error), component="vision_runtime", code="ocr_failed", recoverable=True)

    def _candidate_providers(self, preferred_name: str | None = None) -> list[OCRProvider]:
        preferred = (
            preferred_name,
            self._settings.vision_ocr_provider_default,
            *self._settings.vision_ocr_provider_fallback_order,
        )
        seen: set[str] = set()
        providers: list[OCRProvider] = []
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
