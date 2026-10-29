from __future__ import annotations

from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

import pytest

from jarvis.core.events import EventBus
from jarvis.desktop_runtime.base import DesktopChatMessage
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.desktop_runtime.window import create_qt_application, pyside_available
from jarvis.desktop_runtime.widgets import ConversationSurfaceWidget
from jarvis.runtime_activity import RuntimeActivityStore


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def _fake_desktop_service(event_bus: EventBus, store: RuntimeActivityStore):
    correlation_id = "corr-terminal-ui"
    message_id = str(uuid4())
    service = DesktopRuntimeService.__new__(DesktopRuntimeService)
    service._jarvis = SimpleNamespace(runtime_activity_store=store)
    service._state_lock = RLock()
    service._conversation = [
        DesktopChatMessage(
            message_id=message_id,
            role="assistant",
            content="Respuesta de prueba",
            metadata={"streaming": True},
        )
    ]
    service._progress_message_by_correlation = {correlation_id: message_id}
    service._invalidate_caches = lambda: None
    event_bus.subscribe_all(service._on_runtime_activity)
    return service, correlation_id


def _render(service, app):
    surface = ConversationSurfaceWidget()
    surface.set_messages(service._conversation)
    app.processEvents()
    widgets = surface.message_widgets()
    assert len(widgets) == 1
    thinking = widgets[0]._thinking  # noqa: SLF001 - acceptance test inspects visible widget
    assert thinking is not None
    return surface, thinking


def _publish(event_bus, event_name, correlation_id, **payload):
    event_bus.publish(
        event_name,
        {
            "run_id": "run-terminal-ui",
            "correlation_id": correlation_id,
            "execution_id": "exec-terminal-ui",
            **payload,
        },
    )


def test_terminal_lifecycle_is_rendered_in_real_thinking_widget_without_output_text():
    app = create_qt_application()
    event_bus = EventBus()
    store = RuntimeActivityStore(event_bus)
    service, correlation_id = _fake_desktop_service(event_bus, store)

    _publish(event_bus, "terminal.command_started", correlation_id, command="secret-command")
    surface, thinking = _render(service, app)
    assert "Ejecutando Terminal…" in thinking._details.text()  # noqa: SLF001

    _publish(event_bus, "terminal.output", correlation_id, chunk="SECRET_STDOUT")
    _publish(event_bus, "terminal.output", correlation_id, chunk="SECRET_STDOUT_2")
    surface.set_messages(service._conversation)
    app.processEvents()
    details = thinking._details.text()  # noqa: SLF001
    assert "SECRET_STDOUT" not in details
    assert details.count("Ejecutando Terminal…") == 1

    _publish(event_bus, "terminal.command_completed", correlation_id, exit_code=0)
    surface.set_messages(service._conversation)
    app.processEvents()
    assert "Terminal finalizó correctamente." in thinking._details.text()  # noqa: SLF001
    assert thinking._details.text().index("Ejecutando Terminal…") < thinking._details.text().index(  # noqa: SLF001
        "Terminal finalizó correctamente."
    )

    surface.close()


@pytest.mark.parametrize(
    ("event_name", "expected"),
    [
        ("terminal.command_failed", "Terminal falló."),
        ("terminal.command_timed_out", "Terminal agotó el tiempo."),
        ("terminal.command_cancelled", "Terminal fue cancelada."),
    ],
)
def test_terminal_failure_timeout_and_cancellation_are_distinct_in_thinking(
    event_name, expected
):
    app = create_qt_application()
    event_bus = EventBus()
    store = RuntimeActivityStore(event_bus)
    service, correlation_id = _fake_desktop_service(event_bus, store)

    _publish(event_bus, "terminal.command_started", correlation_id)
    _publish(event_bus, event_name, correlation_id, exit_code=1)
    surface, thinking = _render(service, app)
    details = thinking._details.text()  # noqa: SLF001
    assert expected in details
    if event_name == "terminal.command_timed_out":
        assert "Terminal falló." not in details
    if event_name == "terminal.command_cancelled":
        assert "Terminal agotó el tiempo." not in details
    surface.close()
