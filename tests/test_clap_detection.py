from __future__ import annotations

import array

from jarvis.voice_runtime.base import AudioChunk, AudioEventType
from jarvis.voice_runtime.clap import ClapPatternDetector


def _chunk(level: int) -> AudioChunk:
    return AudioChunk(pcm_bytes=array.array("h", [level] * 1000).tobytes(), duration_seconds=0.1)


def test_clap_detector_resolves_double_and_triple_patterns() -> None:
    detector = ClapPatternDetector(sensitivity=0.5, cooldown_seconds=0.0, window_seconds=2.0)
    detector.process(_chunk(32000), correlation_id="c1")
    detector.process(_chunk(32000), correlation_id="c1")
    double = detector.flush(correlation_id="c1")
    assert double is not None
    assert double.event_type == AudioEventType.LISTENING_START

    detector.process(_chunk(32000), correlation_id="c2")
    detector.process(_chunk(32000), correlation_id="c2")
    triple = detector.process(_chunk(32000), correlation_id="c2")
    assert triple is not None
    assert triple.event_type == AudioEventType.FULL_MODE
