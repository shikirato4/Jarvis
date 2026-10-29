from __future__ import annotations

from datetime import datetime, UTC
from concurrent.futures import Future
import time
from threading import Event
from types import SimpleNamespace

import pytest

from jarvis.desktop_runtime.base import DesktopChatMessage, DesktopChatResponse
from jarvis.desktop_runtime.window import create_qt_application, pyside_available


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def _message(message_id: str, role: str, content: str, *, minute: int) -> DesktopChatMessage:
    return DesktopChatMessage(
        message_id=message_id,
        role=role,
        content=content,
        created_at=datetime(2026, 4, 8, 10, minute, 0, tzinfo=UTC),
    )


def _build_window(tmp_path):
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow
    from jarvis.models_runtime.base import ModelResponse, ProviderKind

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)

    def infer(request):
        return ModelResponse(
            provider_name="nemotron-test",
            provider_kind=ProviderKind.LOCAL,
            logical_model=request.logical_model or "general_assistant",
            model_name="nemotron-test",
            content="Respuesta de Jarvis.",
            latency_ms=1.0,
        )

    backend.agent_controller._infer_model = infer  # noqa: SLF001
    window = JarvisDesktopWindow(desktop)
    return backend, window


def test_conversation_surface_shows_full_transcript(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    try:
        messages = [
            _message("u1", "user", "Abre el canal principal.", minute=0),
            _message("a1", "assistant", "Canal principal abierto.", minute=1),
            _message("u2", "user", "Resume el estado actual.", minute=2),
            _message("a2", "assistant", "Todo opera dentro de parametros normales.", minute=3),
        ]
        window.show()
        window._render_conversation(SimpleNamespace(conversation=messages))  # noqa: SLF001
        app.processEvents()

        assert window._conversation.message_count() == len(messages)  # noqa: SLF001
        assert [widget.text() for widget in window._conversation.message_widgets()] == [m.content for m in messages]  # noqa: SLF001
        assert window._conversation._status.text() == "LIVE"  # noqa: SLF001
    finally:
        window.close()
        backend.stop()


def test_conversation_wraps_long_messages_without_clipping(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    try:
        long_text = " ".join(["Respuesta extendida de JARVIS con multiples segmentos para validar wrapping estable."] * 18)
        messages = [
            _message("u1", "user", "Necesito un informe completo.", minute=0),
            _message("a1", "assistant", long_text, minute=1),
            _message("u2", "user", "Continua.", minute=2),
        ]
        window.resize(1280, 900)
        window.show()
        window._render_conversation(SimpleNamespace(conversation=messages))  # noqa: SLF001
        app.processEvents()

        long_widget = next(widget for widget in window._conversation.message_widgets() if widget.text() == long_text)  # noqa: SLF001
        assert long_widget.height() > 180
        assert long_widget.sizeHint().height() >= long_widget.height() - 8
        assert long_widget.width() <= window._conversation.width()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()


def test_conversation_scrolls_to_latest_message(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    try:
        messages = []
        for index in range(28):
            role = "assistant" if index % 2 else "user"
            content = f"Mensaje {index} " + ("detalle " * 16)
            messages.append(_message(f"m{index}", role, content, minute=index % 59))

        window.resize(1100, 860)
        window.show()
        window._render_conversation(SimpleNamespace(conversation=messages))  # noqa: SLF001
        app.processEvents()

        scrollbar = window._conversation.verticalScrollBar()  # noqa: SLF001
        assert scrollbar.maximum() > 0
        assert scrollbar.value() == scrollbar.maximum()
    finally:
        window.close()
        backend.stop()


def test_conversation_surface_is_wide_and_stable(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    try:
        window.resize(1440, 940)
        window.show()
        app.processEvents()

        assert window._conversation.width() >= 700  # noqa: SLF001
        assert window._conversation.minimumHeight() <= 320  # noqa: SLF001
        assert window._conversation.minimumSizeHint().height() <= 320  # noqa: SLF001
        assert window._conversation.verticalScrollBar().isVisible() is False  # noqa: SLF001
    finally:
        window.close()
        backend.stop()


def test_window_submit_chat_returns_before_background_work_finishes(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    release_background_work = Event()
    try:
        def _slow_send_chat_streaming(text: str, **kwargs):
            release_background_work.wait(timeout=10)
            return DesktopChatResponse(
                message=DesktopChatMessage(message_id="background-complete", role="assistant", content="Trabajo completado."),
                raw_result={"ok": True},
            )

        window._desktop.send_chat_streaming = _slow_send_chat_streaming  # type: ignore[method-assign]  # noqa: SLF001
        window.show()
        window._input.setText("calcula la derivada de x**2")  # noqa: SLF001

        started = time.perf_counter()
        window._submit_chat()  # noqa: SLF001
        elapsed = time.perf_counter() - started
        app.processEvents()

        assert elapsed < 0.15
        assert window._send_button.isEnabled() is False  # noqa: SLF001
        assert window._input.isEnabled() is True  # noqa: SLF001
        release_background_work.set()

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and window._pending_future is not None:  # noqa: SLF001
            window.refresh_view()  # noqa: SLF001
            app.processEvents()
            time.sleep(0.05)

        assert window._pending_future is None  # noqa: SLF001
        assert window._send_button.isEnabled() is True  # noqa: SLF001
    finally:
        release_background_work.set()
        window.close()
        backend.stop()
        window._desktop.shutdown()  # noqa: SLF001


def test_window_keeps_agentic_calculation_on_streaming_path(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    calls = []
    try:
        original = window._desktop.send_chat_async  # noqa: SLF001

        def _capture(text: str, **kwargs):
            calls.append(kwargs.get("metadata"))
            return original(text, **kwargs)

        window._desktop.send_chat_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window.show()
        window._input.setText("calcula 781 por 94")  # noqa: SLF001

        window._submit_chat()  # noqa: SLF001
        app.processEvents()

        assert calls and calls[0]["stream"] is True
    finally:
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and window._pending_future is not None:  # noqa: SLF001
            window.refresh_view()  # noqa: SLF001
            app.processEvents()
            time.sleep(0.05)
        window.close()
        backend.stop()
        window._desktop.shutdown()  # noqa: SLF001


def test_window_submit_chat_clears_input_and_prevents_double_submit(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    calls: list[tuple[str, str | None]] = []
    original = window._desktop.send_chat_async  # noqa: SLF001
    try:
        def _capture(text: str, *, source: str = "text", correlation_id: str | None = None, metadata=None):
            calls.append((text, correlation_id))
            return original(text, source=source, correlation_id=correlation_id, metadata=metadata)

        window._desktop.send_chat_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window.show()
        window._input.setText("hola jarvis")  # noqa: SLF001

        window._submit_chat()  # noqa: SLF001
        first_correlation_id = window._pending_correlation_id  # noqa: SLF001
        assert window._input.text() == ""  # noqa: SLF001

        window._input.setText("hola jarvis")  # noqa: SLF001
        window._submit_chat()  # noqa: SLF001
        app.processEvents()

        assert len(calls) == 1
        assert calls[0][0] == "hola jarvis"
        assert calls[0][1] == first_correlation_id
    finally:
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and window._pending_future is not None:  # noqa: SLF001
            window.refresh_view()  # noqa: SLF001
            app.processEvents()
            time.sleep(0.05)
        window.close()
        backend.stop()
        window._desktop.shutdown()  # noqa: SLF001


def test_deep_research_button_and_slash_command_use_the_same_backend_mode(tmp_path) -> None:
    app = create_qt_application()
    backend, window = _build_window(tmp_path)
    calls: list[tuple[str, dict]] = []

    def _capture(text: str, **kwargs):
        calls.append((text, dict(kwargs.get("metadata") or {})))
        future: Future = Future()
        future.set_result(
            DesktopChatResponse(
                message=DesktopChatMessage(message_id=f"done-{len(calls)}", role="assistant", content="Done [S1]."),
                raw_result={"search_mode": "deep"},
            )
        )
        return future

    try:
        window._desktop.send_chat_async = _capture  # type: ignore[method-assign]  # noqa: SLF001
        window.show()
        window._deep_research_button.setChecked(True)  # noqa: SLF001
        window._input.setText("compare current multimodal models")  # noqa: SLF001
        window._submit_chat()  # noqa: SLF001
        app.processEvents()
        window.refresh_view()
        app.processEvents()

        window._last_submit_at = 0.0  # noqa: SLF001
        window._input.setText("/deepresearch compare current multimodal models")  # noqa: SLF001
        window._submit_chat()  # noqa: SLF001
        app.processEvents()

        assert [item[1]["search_mode"] for item in calls] == ["deep", "deep"]
        assert calls[0][0] == "compare current multimodal models"
        assert calls[1][0].startswith("/deepresearch")
        assert window._deep_research_button.accessibleName() == "Deep Research"  # noqa: SLF001
        assert "Search, read" in window._deep_research_button.toolTip()  # noqa: SLF001
        assert window._stop_research_button.isVisible()  # noqa: SLF001
        assert window._research_activity.isVisible()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        window._desktop.shutdown()  # noqa: SLF001


def test_deep_research_progress_exposes_only_operational_metrics(tmp_path) -> None:
    create_qt_application()
    backend, window = _build_window(tmp_path)
    try:
        backend.event_bus.publish(
            "deep_research.progress",
            {
                "correlation_id": "research-test",
                "stage": "reading",
                "message": "Reading selected sources",
                "sources_discovered": 12,
                "sources_opened": 4,
                "domains": 3,
            },
        )

        progress = window._desktop.research_progress()  # noqa: SLF001

        assert progress["stage"] == "reading"
        assert progress["sources_discovered"] == 12
        assert "prompt" not in progress
        assert "reasoning" not in progress
    finally:
        window.close()
        backend.stop()
        window._desktop.shutdown()  # noqa: SLF001
