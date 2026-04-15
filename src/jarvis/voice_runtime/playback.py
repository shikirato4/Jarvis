from __future__ import annotations

from .backends import AudioOutputRegistry
from .base import CancellationToken, PlaybackHandle, SynthesisResult


class PlaybackController:
    def __init__(self, registry: AudioOutputRegistry, default_backend_name: str) -> None:
        self._registry = registry
        self._default_backend_name = default_backend_name

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        backend = self._select_backend(result)
        if backend is None:
            raise RuntimeError("no audio output backend configured")
        result.backend_name = backend.backend_name
        return backend.play(result, cancellation=cancellation)

    def _select_backend(self, result: SynthesisResult):
        preferred = self._registry.get(self._default_backend_name)
        if preferred is not None:
            if result.audio_bytes and preferred.backend_name in {"winsound", "in_memory"}:
                return preferred
            if result.text_payload and preferred.backend_name in {"pyttsx3", "in_memory"}:
                return preferred
        if result.audio_bytes:
            return self._registry.get("winsound") or self._registry.get("in_memory")
        if result.text_payload:
            return self._registry.get("pyttsx3") or self._registry.get("in_memory")
        return preferred or self._registry.get("in_memory")
