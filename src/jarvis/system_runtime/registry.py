from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .backends import ApplicationCatalogProvider, AssociationProvider, LauncherBackend, VolumeProvider


class VolumeProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, VolumeProvider] = {}

    def register(self, provider: VolumeProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"volume provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str):
        return self._providers.get(provider_name)

    def list_providers(self) -> list[VolumeProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class ApplicationCatalogRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ApplicationCatalogProvider] = {}

    def register(self, provider: ApplicationCatalogProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"application provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str):
        return self._providers.get(provider_name)

    def list_providers(self) -> list[ApplicationCatalogProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class AssociationProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, AssociationProvider] = {}

    def register(self, provider: AssociationProvider) -> None:
        if provider.provider_name in self._providers:
            raise ConfigurationError(f"association provider '{provider.provider_name}' is already registered")
        self._providers[provider.provider_name] = provider

    def get(self, provider_name: str):
        return self._providers.get(provider_name)

    def list_providers(self) -> list[AssociationProvider]:
        return sorted(self._providers.values(), key=lambda item: item.provider_name)


class LauncherBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, LauncherBackend] = {}

    def register(self, backend: LauncherBackend) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(f"launcher backend '{backend.backend_name}' is already registered")
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str):
        return self._backends.get(backend_name)

    def list_backends(self) -> list[LauncherBackend]:
        return sorted(self._backends.values(), key=lambda item: item.backend_name)
