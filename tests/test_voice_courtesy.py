from __future__ import annotations

from types import SimpleNamespace

from jarvis.config import Settings
from jarvis.desktop_runtime.service import DesktopRuntimeService as DesktopService


def _service(tmp_settings: Settings) -> DesktopService:
    service = DesktopService.__new__(DesktopService)
    service._voice = SimpleNamespace(
        speak_literal_calls=[],
        speak_response_calls=[],
        speak_literal=lambda text: service._voice.speak_literal_calls.append(text),
        speak_response=lambda text: service._voice.speak_response_calls.append(text),
    )
    service._sent = []

    def _send_chat(text, *, source="text", correlation_id=None, metadata=None):
        service._sent.append(
            {
                "text": text,
                "source": source,
                "correlation_id": correlation_id,
                "metadata": metadata or {},
            }
        )

    service.send_chat = _send_chat
    service._invalidate_voice_cache = lambda: None
    service._jarvis = SimpleNamespace(settings=tmp_settings)
    return service


def test_voice_thinking_phrase_defaults_on() -> None:
    settings = Settings(_env_file=None)

    assert settings.voice_thinking_phrase == "Un momento, señor."


def test_voice_command_speaks_thinking_phrase_then_sends_to_chat(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_thinking_phrase="Un momento, señor.",
    )
    service = _service(settings)

    service._handle_voice_command("bloquea la pantalla", correlation_id="c-1")

    assert service._voice.speak_literal_calls == ["Un momento, señor."]
    assert len(service._sent) == 1
    assert service._sent[0]["text"] == "bloquea la pantalla"
    assert service._sent[0]["source"] == "voice"
    assert service._sent[0]["metadata"]["surface"] == "desktop_voice"


def test_empty_thinking_phrase_disables_courtesy_speech(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_thinking_phrase="",
    )
    service = _service(settings)

    service._handle_voice_command("hola", correlation_id="c-2")

    assert service._voice.speak_literal_calls == []
    assert len(service._sent) == 1


def test_thinking_phrase_never_carries_reasoning(tmp_path) -> None:
    """The courtesy phrase is fixed text; the model's thoughts are never spoken."""
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        voice_thinking_phrase="Un momento, señor.",
    )
    service = _service(settings)

    service._handle_voice_command("explícame la computación cuántica", correlation_id="c-3")

    spoken = " ".join(service._voice.speak_literal_calls)
    assert "cuántica" not in spoken
    assert spoken == "Un momento, señor."
