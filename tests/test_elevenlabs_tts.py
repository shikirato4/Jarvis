from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.voice_runtime.backend_elevenlabs import ElevenLabsTTSProvider
from jarvis.voice_runtime.base import SynthesisRequest


_KEY = "xi-test-key"
_VOICE = "_voice_21m00Tcm4T2D4"

def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "data_dir": tmp_path / "runtime",
        "workspace_root": tmp_path,
        "elevenlabs_api_key": _KEY,
        "elevenlabs_voice_id": _VOICE,
    }
    values.update(overrides)
    return Settings(**values)


def _client(capture: list, *, status: int = 200, payload: bytes = b"\x01\x00\x02\x00") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        capture.append(request)
        if status != 200:
            return httpx.Response(status, json={"detail": {"message": "invalid_api_key"}})
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_elevenlabs_synthesizes_wav_from_pcm(tmp_path: Path) -> None:
    capture: list = []
    provider = ElevenLabsTTSProvider(
        _settings(tmp_path), client=_client(capture, payload=b"\x01\x00\x02\x00\x03\x00\x04\x00")
    )

    result = provider.synthesize(
        SynthesisRequest(text="Un momento, señor.", play=False)
    )

    assert result.provider_name == "elevenlabs"
    assert result.provider_kind == "cloud"
    assert result.audio_format == "wav"
    assert result.audio_bytes is not None
    assert result.audio_bytes[:4] == b"RIFF"
    assert b"\x01\x00\x02\x00" in result.audio_bytes
    assert result.played is False
    assert result.latency_ms >= 0

    request = capture[0]
    assert request.headers["xi-api-key"] == _KEY
    assert f"/v1/text-to-speech/{_VOICE}" in str(request.url)
    assert "output_format=pcm_44100" in str(request.url)
    import json

    body = json.loads(request.read())
    assert body["text"] == "Un momento, señor."
    assert body["model_id"] == "eleven_multilingual_v2"


def test_elevenlabs_rejects_invalid_credentials_with_clear_error(tmp_path: Path) -> None:
    capture: list = []
    provider = ElevenLabsTTSProvider(_settings(tmp_path), client=_client(capture, status=401))

    with pytest.raises(JarvisError) as excinfo:
        provider.synthesize(SynthesisRequest(text="hola", play=False))

    assert "credencial" in str(excinfo.value).casefold()


def test_elevenlabs_unconfigured_raises_instead_of_calling_the_api(tmp_path: Path) -> None:
    capture: list = []
    provider = ElevenLabsTTSProvider(
        _settings(tmp_path, elevenlabs_api_key=None, elevenlabs_voice_id=""),
        client=_client(capture),
    )
    assert provider.health_check()["healthy"] is False

    with pytest.raises(JarvisError) as excinfo:
        provider.synthesize(SynthesisRequest(text="hola", play=False))

    assert "ELEVENLABS" in str(excinfo.value)
    assert capture == []


def test_bootstrap_registers_elevenlabs_only_with_credentials(tmp_path: Path) -> None:
    from jarvis.bootstrap import build_application

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "ui_backend_kind": "in_memory",
            "system_backend_kind": "in_memory",
            "voice_tts_provider_default": "in_memory",
            "voice_tts_provider_fallback_order": ("in_memory",),
        }
    )
    app = build_application(settings)
    try:
        registry = app.voice_runtime_service._tts._registry  # noqa: SLF001
        assert registry.get("elevenlabs") is not None
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
        assert registry.get("elevenlabs") is None
    finally:
        app2.stop()
