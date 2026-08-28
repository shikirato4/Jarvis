from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .base import LocalBackendAdapter


class LocalBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, LocalBackendAdapter] = {}

    def register(self, backend: LocalBackendAdapter) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(
                f"local backend '{backend.backend_name}' is already registered"
            )
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str) -> LocalBackendAdapter | None:
        return self._backends.get(backend_name)

    def list_backends(self) -> list[LocalBackendAdapter]:
        return sorted(self._backends.values(), key=lambda backend: backend.backend_name)
