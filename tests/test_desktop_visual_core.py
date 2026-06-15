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


def test_window_refresh_skips_full_repaint_when_state_is_unchanged(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    backend, desktop = build_desktop_runtime(settings)
    try:
        window = JarvisDesktopWindow(desktop)
        window.show()
        window.refresh_view()
        first_state = desktop.shell_state()
        applied_before = int(first_state.performance.get("ui_refresh_applied") or 0)
        skipped_before = int(first_state.performance.get("ui_refresh_skipped") or 0)

        window.refresh_view()
        app.processEvents()

        second_state = desktop.shell_state()
        assert int(second_state.performance.get("ui_refresh_applied") or 0) >= applied_before
        assert int(second_state.performance.get("ui_refresh_skipped") or 0) >= skipped_before + 1
    finally:
        backend.stop()
        desktop.shutdown()


def test_stitch_inspired_shell_maps_real_jarvis_surfaces(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
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

        assert set(window._nav_buttons) == {"chat", "agent", "web", "context", "memory", "code", "learning", "settings"}  # noqa: SLF001
        assert window._mode_badge.text().startswith("MODE")  # noqa: SLF001
        assert "gpt-oss" in window._model_badge.text()  # noqa: SLF001
        assert "BLOCKED" in window._openai_badge.text()  # noqa: SLF001
        assert "BLOCKED" in window._gemini_badge.text()  # noqa: SLF001
        assert window._right_tabs.count() >= 7  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_agent_mode_is_visible_and_guided_controls_are_available(tmp_path) -> None:
    from jarvis.config import Settings
    from jarvis.desktop import build_desktop_runtime
    from jarvis.desktop_runtime.window import JarvisDesktopWindow

    app = create_qt_application()
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
        window._navigate_shell("agent")  # noqa: SLF001
        app.processEvents()

        assert window._right_panel.isVisible()  # noqa: SLF001
        assert window._right_tabs.tabText(window._right_tabs.currentIndex()) == "Agent Mode"  # noqa: SLF001
        assert window._start_guided_agent_button.isEnabled()  # noqa: SLF001
        assert not window._confirm_action_button.isEnabled()  # noqa: SLF001
        assert not window._stop_agent_button.isEnabled()  # noqa: SLF001
        assert "Guided Control" in window._agent_note.text()  # noqa: SLF001
    finally:
        window.close()
        backend.stop()
        desktop.shutdown()


def test_reactor_accepts_future_visual_states() -> None:
    from jarvis.desktop_runtime.widgets import ReactorCoreWidget

    reactor = ReactorCoreWidget()
    for state in ("idle", "listening", "thinking", "speaking", "web_search", "agent_preview"):
        reactor.set_state(state, activity=0.5)
    assert reactor.sizeHint().width() >= 200
