from __future__ import annotations

import array
import time
from types import SimpleNamespace

import pytest

from jarvis.config import Settings
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_runtime.base import DesktopVoiceState
from jarvis.desktop_runtime.chat import PendingDesktopAction
from jarvis.desktop_runtime.intent_router import DesktopIntentDecision
from jarvis.desktop_runtime.window import JarvisDesktopWindow, create_qt_application, pyside_available
from jarvis.voice_runtime.backends import InMemorySTTProvider
from jarvis.voice_runtime.base import AudioChunk


def _chunk(text: str, level: int = 8000) -> AudioChunk:
    pcm = array.array("h", [level] * 2000).tobytes()
    return AudioChunk(pcm_bytes=pcm, duration_seconds=0.2, metadata={"mock_text": text})


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_audio_input_backend_default="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_stt_provider_default="in_memory",
        voice_tts_provider_default="in_memory",
        voice_input_provider_default="in_memory",
        voice_input_timeout_seconds=1.0,
    )


def _wait_for(predicate, *, timeout: float = 12.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_desktop_voice_input_routes_transcript_through_existing_chat_pipeline(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("calcula derivada de x^2"))  # noqa: SLF001
        started = time.perf_counter()
        desktop.start_voice_listening()
        assert (time.perf_counter() - started) < 0.2
        assert _wait_for(lambda: any(message.metadata.get("source") == "voice" for message in desktop.shell_state().conversation))
        assert _wait_for(lambda: desktop.shell_state().voice.input_state == "IDLE")
        state = desktop.shell_state()
        assert any(message.metadata.get("source") == "voice" for message in state.conversation)
        assert state.voice.last_transcript == "calcula derivada de x^2"
        assert state.voice.input_state == "IDLE"
        assert "deriv" in state.conversation[-1].content.casefold() or "dos equis" in state.conversation[-1].content.casefold()
    finally:
        app.stop()


def test_desktop_voice_input_accepts_voice_confirmation(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        desktop._chat.pending_action = PendingDesktopAction(  # noqa: SLF001
            decision=DesktopIntentDecision(category="system_open", prompt="abre word", target="word"),
            prompt="abre word",
        )
        desktop._chat.awaiting_confirmation = True  # noqa: SLF001
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("si"))  # noqa: SLF001
        desktop.start_voice_listening()
        assert _wait_for(lambda: desktop._chat.awaiting_confirmation is False)  # noqa: SLF001
        assert _wait_for(lambda: any("Acceso concedido" in message.content for message in desktop.shell_state().conversation))
    finally:
        app.stop()


def test_desktop_voice_input_executes_open_and_type_flow(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    backend = app.ui_automation_service._backend  # noqa: SLF001
    try:
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("abre word y escribe esto hola por voz"))  # noqa: SLF001
        desktop.start_voice_listening()
        assert _wait_for(lambda: backend.typed_text.endswith("hola por voz"))
        assert _wait_for(lambda: any(message.metadata.get("source") == "voice" for message in desktop.shell_state().conversation))
        state = desktop.shell_state()
        assert any(message.metadata.get("source") == "voice" for message in state.conversation)
        assert backend.get_active_window().title == "Word"
    finally:
        app.stop()


def test_desktop_voice_input_surfaces_stt_error_state(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        app.voice_runtime_service._stt._registry._providers["in_memory"] = InMemorySTTProvider(fail=True)  # noqa: SLF001
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("abre word"))  # noqa: SLF001
        desktop.start_voice_listening()
        assert _wait_for(lambda: desktop.shell_state().voice.input_state == "ERROR")
        state = desktop.shell_state()
        assert state.voice.input_error
        assert "failed" in state.voice.input_error.casefold()
    finally:
        app.stop()


def test_desktop_voice_input_reports_empty_audio(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        desktop.start_voice_listening()
        assert _wait_for(lambda: desktop.shell_state().voice.input_state == "ERROR")
        state = desktop.shell_state()
        assert "no speech detected" in (state.voice.input_error or "")
    finally:
        app.stop()


@pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")
def test_window_reflects_voice_input_states(tmp_path) -> None:
    qt_app = create_qt_application()
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    window = JarvisDesktopWindow(desktop)
    try:
        state = SimpleNamespace(
            panel_snapshot=SimpleNamespace(health_summary={"aggregate_status": "ready", "active_operations": 0}, alerts=[]),
            conversation=[],
            voice=DesktopVoiceState(
                enabled=True,
                muted=False,
                speaking=False,
                input_enabled=True,
                input_muted=False,
                input_state="LISTENING",
                input_error=None,
                last_transcript=None,
            ),
        )
        window._render_header(state)  # noqa: SLF001
        window._render_conversation(state)  # noqa: SLF001
        window._sync_reactor_state(state)  # noqa: SLF001
        qt_app.processEvents()

        assert window._voice_badge.text() == "VOICE LISTENING"  # noqa: SLF001
        assert window._conversation._status.text() == "LISTENING"  # noqa: SLF001

        state.voice.input_state = "ERROR"
        state.voice.input_error = "Micrófono no disponible."
        window._render_header(state)  # noqa: SLF001
        window._render_conversation(state)  # noqa: SLF001
        qt_app.processEvents()

        assert window._voice_badge.text() == "VOICE ERROR"  # noqa: SLF001
        assert window._conversation._meta.text() == "Micrófono no disponible."  # noqa: SLF001
    finally:
        window.close()
        app.stop()
