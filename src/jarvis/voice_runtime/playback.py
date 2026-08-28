from __future__ import annotations

from .backends import AudioOutputRegistry
from .base import AudioOutputBackend, CancellationToken, PlaybackHandle, SynthesisResult


class PlaybackController:
    def __init__(self, registry: AudioOutputRegistry, default_backend_name: str) -> None:
        self._registry = registry
        self._default_backend_name = default_backend_name

    def play(self, result: SynthesisResult, *, cancellation: CancellationToken | None = None) -> PlaybackHandle:
        candidates = self._candidate_backends(result)
        if not candidates:
            raise RuntimeError("no audio output backend configured")
        failures: list[tuple[str, Exception]] = []
        for backend in candidates:
            result.backend_name = backend.backend_name
            try:
                handle = backend.play(result, cancellation=cancellation)
            except Exception as exc:  # noqa: BLE001
                failures.append((backend.backend_name, exc))
                continue
            if failures:
                failed_backend, failure = failures[-1]
                result.metadata["playback_fallback_from"] = failed_backend
                result.metadata["playback_fallback_reason"] = str(failure)
            return handle
        raise failures[-1][1]

    def _candidate_backends(self, result: SynthesisResult) -> list[AudioOutputBackend]:
        candidates: list[AudioOutputBackend] = []

        def add(backend: AudioOutputBackend | None) -> None:
            if backend is not None and all(candidate.backend_name != backend.backend_name for candidate in candidates):
                candidates.append(backend)

        preferred = self._registry.get(self._default_backend_name)
        if preferred is not None:
            if result.audio_bytes and preferred.backend_name in {"winsound", "in_memory"}:
                add(preferred)
            if result.text_payload and preferred.backend_name in {"pyttsx3", "in_memory"}:
                add(preferred)
        if result.audio_bytes:
            add(self._registry.get("winsound"))
            add(self._registry.get("in_memory"))
        if result.text_payload:
            add(self._registry.get("pyttsx3"))
            add(self._registry.get("in_memory"))
        if not candidates:
            add(preferred)
            add(self._registry.get("in_memory"))
        return candidates
