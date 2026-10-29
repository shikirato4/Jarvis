from __future__ import annotations

import io
from pathlib import Path

import numpy
import pytest
import soundfile as sf

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.voice_runtime.backend_edge_tts import EdgeTTSProvider
from jarvis.voice_runtime.base import SynthesisRequest


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "data_dir": tmp_path / "runtime",
        "workspace_root": tmp_path,
        "voice_edge_tts_enabled": True,
        "voice_edge_tts_voice": "es-MX-JorgeNeural",
    }
    values.update(overrides)
    return Settings(**values)


def _tiny_mp3() -> bytes:
    """Real MP3 generated in-memory so tests never touch the network."""
    rate = 24_000
    t = numpy.linspace(0, 0.3, int(rate * 0.3), endpoint=False)
    tone = (0.3 * numpy.sin(2 * numpy.pi * 440 * t)).astype("float32")
    buffer = io.BytesIO()
    sf.write(buffer, tone, rate, format="MP3")
    return buffer.getvalue()


class _FakeCommunicate:
    calls: list[dict] = []

    def __init__(self, text: str, voice: str, **kwargs) -> None:
        self._chunks = [
            {"type": "audio", "data": _tiny_mp3()},
            {"type": "WordBoundary", "offset": 1, "duration": 1, "text": "x"},
        ]
        _FakeCommunicate.calls.append({"text": text, "voice": voice})

    def stream(self):
        async def _generator():
            for chunk in self._chunks:
                yield chunk

        return _generator()


def _provider(settings: Settings) -> EdgeTTSProvider:
    return EdgeTTSProvider(settings, communicate_factory=_FakeCommunicate)


def test_edge_tts_synthesizes_playable_wav_from_cloud_mp3(tmp_path: Path) -> None:
    _FakeCommunicate.calls.clear()
    provider = _provider(_settings(tmp_path))

    result = provider.synthesize(SynthesisRequest(text="Buenos días, señor.", play=False))

    assert result.provider_name == "edge_tts"
    assert result.provider_kind == "cloud"
    assert result.audio_format == "wav"
    assert result.audio_bytes is not None
    assert result.audio_bytes[:4] == b"RIFF"
    assert len(result.audio_bytes) > 1000
    assert _FakeCommunicate.calls == [
        {"text": "Buenos días, señor.", "voice": "es-MX-JorgeNeural"}
    ]


def test_edge_tts_disabled_raises_instead_of_streaming(tmp_path: Path) -> None:
    _FakeCommunicate.calls.clear()
    provider = _provider(_settings(tmp_path, voice_edge_tts_enabled=False))

    with pytest.raises(JarvisError) as excinfo:
        provider.synthesize(SynthesisRequest(text="hola", play=False))

    assert "EDGE_TTS" in str(excinfo.value) or "deshabilitado" in str(excinfo.value).casefold()
    assert _FakeCommunicate.calls == []


def test_edge_tts_health_reflects_configuration(tmp_path: Path) -> None:
    enabled = _provider(_settings(tmp_path)).health_check()
    disabled = _provider(_settings(tmp_path, voice_edge_tts_enabled=False)).health_check()

    assert enabled["healthy"] is True
    assert disabled["healthy"] is False


def test_bootstrap_registers_edge_tts_only_when_enabled(tmp_path: Path) -> None:
    from jarvis.bootstrap import build_application

    settings = _settings(
        tmp_path,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_tts_provider_default="in_memory",
        voice_tts_provider_fallback_order=("in_memory",),
    )
    app = build_application(settings)
    try:
        registry = app.voice_runtime_service._tts._registry  # noqa: SLF001
        assert registry.get("edge_tts") is not None
    finally:
        app.stop()

    bare = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime2",
        workspace_root=tmp_path,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_tts_provider_default="in_memory",
        voice_tts_provider_fallback_order=("in_memory",),
    )
    app2 = build_application(bare)
    try:
        registry = app2.voice_runtime_service._tts._registry  # noqa: SLF001
        assert registry.get("edge_tts") is None
    finally:
        app2.stop()
