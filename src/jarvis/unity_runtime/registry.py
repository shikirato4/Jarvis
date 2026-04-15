from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .backends import UnityInstallationProvider
from .bridge import UnityBridgeBackend


class UnityInstallationRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, UnityInstallationProvider] = {}

    def register(self, provider: UnityInstallationProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"unity installation provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def list_providers(self) -> list[UnityInstallationProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class UnityBridgeRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, UnityBridgeBackend] = {}

    def register(self, backend: UnityBridgeBackend) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(f"unity bridge backend '{backend.backend_name}' is already registered")
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str):
        return self._backends.get(backend_name)

    def list_backends(self) -> list[UnityBridgeBackend]:
        return sorted(self._backends.values(), key=lambda item: item.backend_name)
