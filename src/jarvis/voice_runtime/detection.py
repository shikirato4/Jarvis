from __future__ import annotations

import array

from .base import AudioChunk, VoiceDetectionResult


def detect_audio_activity(
    chunk: AudioChunk,
    *,
    silence_threshold: float,
    speech_threshold: float,
) -> VoiceDetectionResult:
    if not chunk.pcm_bytes:
        return VoiceDetectionResult(silence_detected=True)
    samples = array.array("h")
    samples.frombytes(chunk.pcm_bytes)
    if not samples:
        return VoiceDetectionResult(silence_detected=True)
    energies = [abs(sample) / 32768 for sample in samples]
    average = sum(energies) / len(energies)
    peak = max(energies)
    return VoiceDetectionResult(
        speech_detected=average >= speech_threshold,
        silence_detected=average <= silence_threshold,
        average_energy=average,
        peak_energy=peak,
    )
