from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .base import ScreenCaptureBackend


class ScreenCaptureBackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, ScreenCaptureBackend] = {}

    def register(self, backend: ScreenCaptureBackend) -> None:
        if backend.backend_name in self._backends:
            raise ConfigurationError(f"screen capture backend '{backend.backend_name}' is already registered")
        self._backends[backend.backend_name] = backend

    def get(self, backend_name: str) -> ScreenCaptureBackend | None:
        return self._backends.get(backend_name)

    def list_backends(self) -> list[ScreenCaptureBackend]:
        return sorted(self._backends.values(), key=lambda item: item.backend_name)
