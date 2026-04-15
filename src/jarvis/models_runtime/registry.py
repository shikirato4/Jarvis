from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .base import ModelProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}

    def register(self, provider: ModelProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str) -> ModelProvider | None:
        return self._providers.get(provider_name)

    def list_providers(self) -> list[ModelProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)
