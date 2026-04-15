from __future__ import annotations

from .base import AudioBufferWindow, AudioChunk


class AudioWindowBuffer:
    def __init__(self) -> None:
        self._chunks: list[AudioChunk] = []
        self._duration = 0.0

    def append(self, chunk: AudioChunk) -> None:
        self._chunks.append(chunk)
        self._duration += chunk.duration_seconds

    def extend(self, chunks: list[AudioChunk]) -> None:
        for chunk in chunks:
            self.append(chunk)

    def clear(self) -> None:
        self._chunks.clear()
        self._duration = 0.0

    def snapshot(self) -> AudioBufferWindow:
        return AudioBufferWindow(chunks=list(self._chunks), total_duration_seconds=self._duration)

    def merged_pcm(self) -> bytes:
        return b"".join(chunk.pcm_bytes for chunk in self._chunks)

    def chunks(self) -> list[AudioChunk]:
        return list(self._chunks)
