from __future__ import annotations

from datetime import datetime, timezone
import time
from types import SimpleNamespace

import pytest

from jarvis.desktop_runtime.base import DesktopChatMessage
from jarvis.desktop_runtime.window import create_qt_application, pyside_available


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def _message(message_id: str, role: str, content: str, *, minute: int) -> DesktopChatMessage:
    return DesktopChatMessage(
        message_id=message_id,
        role=role,
        content=content,
        created_at=datetime(2026, 4, 8, 10, minute, 0, tzinfo=timezone.utc),
    )


def _build_window(tmp_path):
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
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
    original = window._desktop.send_chat  # noqa: SLF001
    try:
        def _slow_send_chat(text: str, **kwargs):
            time.sleep(0.3)
            return original(text, **kwargs)

        window._desktop.send_chat = _slow_send_chat  # type: ignore[method-assign]  # noqa: SLF001
        window.show()
        window._input.setText("calcula la derivada de x**2")  # noqa: SLF001

        started = time.perf_counter()
        window._submit_chat()  # noqa: SLF001
        elapsed = time.perf_counter() - started
        app.processEvents()

        assert elapsed < 0.15
        assert window._send_button.isEnabled() is False  # noqa: SLF001
        assert window._input.isEnabled() is True  # noqa: SLF001

        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline and window._pending_future is not None:  # noqa: SLF001
            window.refresh_view()  # noqa: SLF001
            app.processEvents()
            time.sleep(0.05)

        assert window._pending_future is None  # noqa: SLF001
        assert window._send_button.isEnabled() is True  # noqa: SLF001
    finally:
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
