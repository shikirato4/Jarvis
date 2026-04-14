import time

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import WritingRuntimeError
from jarvis.writing_runtime.models import WritingContinuationRequest, WritingGeneratedBlock


def test_writing_continuation_writes_into_active_word_window(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="writing continue")
    jarvis_app.runtime_service.ui_focus_window({"target": "Word"})
    backend = jarvis_app.ui_automation_service._backend  # noqa: SLF001
    backend.typed_text = "Sergio miró la estación vacía y recordó la promesa."
    receipt = jarvis_app.runtime_service.writing_continue(
        WritingContinuationRequest(
            prompt="continúa la escena",
            target_window="Word",
            desired_words=60,
            write_directly=True,
        )
    )
    assert receipt.success is True
    assert receipt.written_text
    assert receipt.written_text in backend.typed_text


def test_writing_continuation_uses_phase_specific_ui_write_timeout(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        ui_watchdog_timeout_ms=500,
        writing_ui_write_timeout_ms=3_000,
        writing_ui_typing_interval_ms=2,
        writing_ui_block_size=500,
        writing_ui_pause_between_blocks_ms=0,
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="writing ui timeout split")
        app.runtime_service.ui_focus_window({"target": "Word"})
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend.typed_text = "Contexto narrativo suficiente para continuar la historia en Word. " * 4
        app.writing_runtime_service._continuation.continue_text = lambda *args, **kwargs: WritingGeneratedBlock(  # noqa: SLF001
            index=1,
            text="A" * 600,
            word_count=120,
            confidence=0.8,
            style_notes=[],
        )

        receipt = app.runtime_service.writing_continue(
            WritingContinuationRequest(prompt="continua con el mismo tono", target_window="Word", desired_words=120, write_directly=True)
        )

        assert receipt.success is True
        assert receipt.written_text == "A" * 600
        events = app.event_bus.recent_events()
        started = [event for event in events if event["event_name"] == "ops.operation.started"]
        assert any(event["payload"]["operation_name"] == "writing.context" for event in started)
        assert any(event["payload"]["operation_name"] == "writing.generation" for event in started)
        assert any(event["payload"]["operation_name"] == "writing.ui_write" for event in started)
    finally:
        app.stop()


def test_writing_timeout_reports_generation_phase_clearly(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        writing_generation_timeout_ms=20,
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="writing generation timeout")
        app.runtime_service.ui_focus_window({"target": "Word"})
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend.typed_text = "Contexto suficiente para disparar la fase de generacion. " * 3

        def _slow_generation(*args, **kwargs):
            time.sleep(0.05)
            return WritingGeneratedBlock(index=1, text="Texto lento", word_count=2, confidence=0.5, style_notes=[])

        app.writing_runtime_service._continuation.continue_text = _slow_generation  # noqa: SLF001

        with pytest.raises(WritingRuntimeError, match="La generación tardó demasiado."):
            app.runtime_service.writing_continue(
                WritingContinuationRequest(prompt="continua la historia", target_window="Word", desired_words=60, write_directly=True)
            )
    finally:
        app.stop()


def test_writing_timeout_reports_ui_write_phase_clearly(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        writing_ui_write_timeout_ms=20,
    )
    app = build_application(settings)
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="writing ui timeout")
        app.runtime_service.ui_focus_window({"target": "Word"})
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend.typed_text = "Contexto suficiente para escritura real. " * 3
        app.writing_runtime_service._continuation.continue_text = lambda *args, **kwargs: WritingGeneratedBlock(  # noqa: SLF001
            index=1,
            text="Texto para escribir",
            word_count=3,
            confidence=0.8,
            style_notes=[],
        )

        original_write = app.writing_runtime_service._editor.write  # noqa: SLF001

        def _slow_write(*args, **kwargs):
            time.sleep(0.05)
            return original_write(*args, **kwargs)

        app.writing_runtime_service._editor.write = _slow_write  # noqa: SLF001

        with pytest.raises(WritingRuntimeError, match="La escritura en Word excedió el tiempo permitido."):
            app.runtime_service.writing_continue(
                WritingContinuationRequest(prompt="continua la historia", target_window="Word", desired_words=60, write_directly=True)
            )
    finally:
        app.stop()
