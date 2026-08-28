from pathlib import Path

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings


@pytest.fixture(autouse=True)
def isolate_optional_audio_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from local audio hardware and user .env choices."""
    monkeypatch.setenv("JARVIS_VOICE_AUDIO_INPUT_BACKEND_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_AUDIO_OUTPUT_BACKEND_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_TTS_PROVIDER_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_TTS_PROVIDER_FALLBACK_ORDER", '["in_memory"]')


@pytest.fixture()
def jarvis_app(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        command_allowlist=("where",),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        yield app
    finally:
        app.stop()
