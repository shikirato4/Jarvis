from __future__ import annotations

import pytest

from jarvis.desktop_runtime.window import create_qt_application, pyside_available


pytestmark = pytest.mark.skipif(not pyside_available(), reason="PySide6 not available")


def test_reactor_widget_and_window_boot(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.widgets import ReactorCoreWidget
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    reactor = ReactorCoreWidget()
    reactor.set_state("active", activity=0.8)
    assert reactor.sizeHint().width() > 0
    assert reactor.minimumSizeHint().height() <= 220

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
    try:
        window = JarvisDesktopWindow(desktop)
        assert window._reactor is not None  # noqa: SLF001
        assert window._conversation is not None  # noqa: SLF001
        assert window._input.placeholderText()  # noqa: SLF001
        assert window.minimumHeight() <= 720  # noqa: SLF001
        assert window.minimumWidth() <= 1280  # noqa: SLF001
        assert window._conversation.minimumHeight() <= 320  # noqa: SLF001
        assert window._focus_mode is True  # noqa: SLF001
        assert window._left_panel.isHidden()  # noqa: SLF001
        assert window._right_panel.isHidden()  # noqa: SLF001
        window._set_focus_mode(False)  # noqa: SLF001
        assert not window._left_panel.isHidden()  # noqa: SLF001
        assert not window._right_panel.isHidden()  # noqa: SLF001
        window.refresh_view()
        app.processEvents()
    finally:
        backend.stop()


def test_window_geometry_fits_available_screen(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    screen = app.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry()

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
    try:
        window = JarvisDesktopWindow(desktop)
        window.show()
        app.processEvents()

        frame = window.frameGeometry()
        assert frame.width() <= available.width()
        assert frame.height() <= available.height()
        assert window.minimumSizeHint().height() <= available.height()
    finally:
        backend.stop()


def test_window_geometry_defaults_are_safe_for_full_hd(tmp_path, monkeypatch) -> None:
    from PySide6.QtCore import QRect
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    monkeypatch.setattr(JarvisDesktopWindow, "_available_geometry", lambda self: QRect(0, 0, 1920, 1080))

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
    try:
        window = JarvisDesktopWindow(desktop)
        window.show()
        app.processEvents()

        frame = window.frameGeometry()
        assert frame.width() <= 1920
        assert frame.height() <= 1080
        assert window.width() <= 1600
        assert window.height() <= 900
        assert window.minimumWidth() <= 1280
        assert window.minimumHeight() <= 720
    finally:
        backend.stop()
