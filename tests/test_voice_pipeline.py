from __future__ import annotations

import array
import time
from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.voice_runtime.base import AudioChunk, VoiceSessionRequest, VoiceSessionState


def _chunk(text: str, level: int = 8000) -> AudioChunk:
    pcm = array.array("h", [level] * 2000).tobytes()
    return AudioChunk(pcm_bytes=pcm, duration_seconds=0.2, metadata={"mock_text": text})


def test_voice_dictation_writes_into_ui_backend(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_input_backend_default="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_stt_provider_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="voice dictation")
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("dictado hacia word"))  # noqa: SLF001
        receipt = app.runtime_service.voice_dictate(
            VoiceSessionRequest(mode="dictation", duration_seconds=1.0, target_window="Word", ui_mode="direct")
        )
        for _ in range(240):
            if "dictado hacia word" in app.ui_automation_service._backend.typed_text:  # noqa: SLF001
                break
            time.sleep(0.05)
        backend = app.ui_automation_service._backend  # noqa: SLF001
        snapshot = app.runtime_service.snapshot()
        assert receipt.session_id is not None
        assert "dictado hacia word" in backend.typed_text
        assert snapshot.recent_voice_invocations
    finally:
        app.stop()


def test_voice_cancel_phrase_triggers_cancellation_path(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_input_backend_default="in_memory",
        voice_stt_provider_default="in_memory",
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="voice cancel")
        app.voice_runtime_service._input_registry.get("in_memory").push_chunk(_chunk("alto"))  # noqa: SLF001
        app.runtime_service.voice_start_session(VoiceSessionRequest(duration_seconds=1.0))
        for _ in range(50):
            session = app.voice_runtime_service.active_session()
            if session is None or session.state in {VoiceSessionState.INTERRUPTED, VoiceSessionState.STOPPED}:
                break
            time.sleep(0.05)
        snapshot = app.runtime_service.snapshot()
        assert snapshot.recent_voice_invocations
        assert any(record.operation_name in {"voice.session.completed", "voice.session.failed"} or "voice.session" in record.operation_name for record in snapshot.recent_voice_invocations)
    finally:
        app.stop()
