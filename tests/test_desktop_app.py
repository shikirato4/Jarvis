from __future__ import annotations

import time

from jarvis.desktop import build_desktop_runtime
from jarvis.config import Settings
from jarvis.desktop_runtime.panels import DesktopPanelComposer


def test_desktop_runtime_boot_and_panels(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        state = desktop.shell_state()
        assert state.app_name == "Jarvis"
        assert state.quick_actions
        composer = DesktopPanelComposer(desktop.bridge())
        snapshot = composer.compose()
        assert snapshot.services
        assert "aggregate_status" in snapshot.health_summary
    finally:
        app.stop()


def test_desktop_runtime_chat_panel_remains_functional(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("calcula la derivada de x**2")
        assert response.message.content
        assert "orchestrated" not in response.message.content
    finally:
        app.stop()


def test_desktop_runtime_routes_clean_final_text_to_voice(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    captured: list[tuple[str, str | None]] = []
    try:
        original = app.voice_runtime_service.speak

        def _capture(text: str, *, correlation_id: str | None = None):
            captured.append((text, correlation_id))
            return original(text, correlation_id=correlation_id)

        app.voice_runtime_service.speak = _capture  # type: ignore[method-assign]
        response = desktop.send_chat("calcula la derivada de x**2")
        assert captured
        assert captured[-1][0] == response.spoken_content
        assert response.spoken_content != response.message.content
        assert "dos equis" in captured[-1][0].casefold()
        assert "metadata" not in captured[-1][0].casefold()
    finally:
        app.stop()


def test_desktop_runtime_exposes_voice_controls(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        state = desktop.shell_state()
        assert state.voice.enabled is True
        desktop.set_voice_muted(True)
        muted_state = desktop.shell_state()
        assert muted_state.voice.muted is True
        desktop.set_voice_enabled(False)
        disabled_state = desktop.shell_state()
        assert disabled_state.voice.enabled is False
    finally:
        app.stop()


def test_desktop_runtime_can_defer_backend_startup(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings, start=False)
    try:
        assert app.started is False
        initial = desktop.shell_state()
        assert initial.busy is True
        assert initial.activity_label == "STARTING"

        desktop.start_backend()

        ready = desktop.shell_state()
        assert app.started is True
        assert ready.activity_label == "IDLE"
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_shell_state_reuses_cached_panel_snapshot(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    calls = 0
    original = desktop._panels.compose  # noqa: SLF001
    try:
        def _compose():
            nonlocal calls
            calls += 1
            return original()

        desktop._panels.compose = _compose  # type: ignore[method-assign]  # noqa: SLF001
        desktop.shell_state()
        desktop.shell_state()
        assert calls == 1
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_send_chat_async_sets_busy_without_blocking(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    original = desktop._chat.handle  # noqa: SLF001
    try:
        def _slow_handle(text: str):
            time.sleep(0.25)
            return original(text)

        desktop._chat.handle = _slow_handle  # type: ignore[method-assign]  # noqa: SLF001
        started = time.perf_counter()
        future = desktop.send_chat_async("calcula la derivada de x**2")
        elapsed = time.perf_counter() - started
        assert elapsed < 0.1
        assert desktop.shell_state().busy is True
        response = future.result(timeout=5.0)
        assert response.message.content
        final_state = desktop.shell_state()
        assert final_state.busy is False
        assert final_state.performance["last_request_ms"] is not None
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_send_chat_async_deduplicates_same_correlation_id(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    calls = 0
    original = desktop._chat.handle  # noqa: SLF001
    try:
        def _slow_handle(text: str):
            nonlocal calls
            calls += 1
            time.sleep(0.2)
            return original(text)

        desktop._chat.handle = _slow_handle  # type: ignore[method-assign]  # noqa: SLF001
        future_a = desktop.send_chat_async("calcula la derivada de x**2", correlation_id="same-request")
        future_b = desktop.send_chat_async("calcula la derivada de x**2", correlation_id="same-request")
        assert future_a is future_b
        future_a.result(timeout=5.0)
        assert calls == 1
        user_messages = [message for message in desktop.shell_state().conversation if message.role == "user"]
        assert len(user_messages) == 1
    finally:
        app.stop()
        desktop.shutdown()
