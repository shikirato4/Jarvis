from __future__ import annotations

import array

from jarvis.voice_runtime.base import AudioChunk, AudioEventType
from jarvis.voice_runtime.clap import ClapPatternDetector
from jarvis.voice_runtime.detection import detect_audio_activity


def _pcm_chunk(value: int, *, samples: int = 2000) -> AudioChunk:
    pcm = array.array("h", [value] * samples).tobytes()
    return AudioChunk(pcm_bytes=pcm, duration_seconds=0.2)


def test_audio_detection_flags_silence_and_speech() -> None:
    silence = detect_audio_activity(_pcm_chunk(0), silence_threshold=0.02, speech_threshold=0.05)
    speech = detect_audio_activity(_pcm_chunk(5000), silence_threshold=0.02, speech_threshold=0.05)
    assert silence.silence_detected is True
    assert speech.speech_detected is True


def test_audio_events_map_claps_to_safe_signals() -> None:
    detector = ClapPatternDetector(sensitivity=0.4, cooldown_seconds=0.0, window_seconds=1.0)
    first = detector.process(_pcm_chunk(30000), correlation_id="clap-1")
    assert first is None
    second = detector.flush(correlation_id="clap-1")
    assert second is not None
    assert second.event_type == AudioEventType.ATTENTION
