from __future__ import annotations

import struct

import pytest

from jarvis.voice_runtime.buffers import wav_duration_seconds
from jarvis.voice_runtime.backend_edge_tts import EdgeTTSProvider
from jarvis.voice_runtime.backend_elevenlabs import _pcm_to_wav
from jarvis.voice_runtime.base import SynthesisRequest

from tests.test_edge_tts import _settings, _FakeCommunicate


def _wav(rate: int, channels: int, bits: int, data_bytes: int) -> bytes:
    header = bytearray(b"RIFF")
    header += struct.pack("<I", 36 + data_bytes)
    header += b"WAVEfmt "
    header += struct.pack(
        "<IHHIIHH", 16, 1, channels, rate,
        rate * channels * bits // 8, channels * bits // 8, bits,
    )
    header += b"data"
    header += struct.pack("<I", data_bytes)
    return bytes(header) + (b"\x00" * data_bytes)


def test_wav_duration_seconds_parses_standard_header() -> None:
    # 24000 Hz, mono, 16-bit, 4800 bytes of data = 0.1 s
    wav = _wav(24000, 1, 16, 4800)
    assert wav_duration_seconds(wav) == pytest.approx(0.1, abs=1e-6)


def test_wav_duration_seconds_stereo_and_other_rates() -> None:
    # 0.5 s of stereo 16-bit at 44100 Hz = 44100 frames/s * 0.5 * 4 bytes
    assert wav_duration_seconds(_wav(44100, 2, 16, 88200)) == pytest.approx(0.5)
    assert wav_duration_seconds(_wav(16000, 1, 16, 32000)) == pytest.approx(1.0)


def test_wav_duration_seconds_safe_on_garbage() -> None:
    assert wav_duration_seconds(b"") == 0.0
    assert wav_duration_seconds(b"RIFF") == 0.0
    assert wav_duration_seconds(b"NOPE" + b"x" * 100) == 0.0


def test_edge_tts_result_reports_real_audio_duration(tmp_path) -> None:
    provider = EdgeTTSProvider(_settings(tmp_path), communicate_factory=_FakeCommunicate)
    result = provider.synthesize(SynthesisRequest(text="hola", play=False))

    # The fixture is a 0.3 s tone; before the fix this stayed at the
    # 3.0 s default and long-text segments overlapped on playback.
    assert result.duration_seconds == pytest.approx(0.3, abs=0.05)


def test_pcm_to_wav_feed_yields_exact_duration() -> None:
    wav = _pcm_to_wav(b"\x01\x00" * 24000, sample_rate=24000)
    assert wav_duration_seconds(wav) == pytest.approx(1.0, abs=1e-6)
