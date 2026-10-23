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


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Duration of a standard 44-byte-header RIFF/WAV file.

    Providers that wrap PCM into WAV locally use this to report the real
    audio length: the segment player waits on this value, so a wrong
    duration made long texts overlap (the old fixed 3.0 s default).
    Returns 0.0 for anything that is not a parseable WAV.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return 0.0
    import struct

    channels = struct.unpack("<H", wav_bytes[22:24])[0]
    sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
    bits_per_sample = struct.unpack("<H", wav_bytes[34:36])[0]
    data_size = struct.unpack("<I", wav_bytes[40:44])[0]
    if sample_rate <= 0 or channels <= 0 or bits_per_sample < 8:
        return 0.0
    bytes_per_frame = channels * (bits_per_sample // 8)
    if bytes_per_frame <= 0:
        return 0.0
    return data_size / (sample_rate * bytes_per_frame)
