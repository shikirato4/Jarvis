from __future__ import annotations

import time

from jarvis.desktop import build_desktop_runtime
from jarvis.models_runtime.base import StreamChunk
from jarvis.config import Settings
from jarvis.desktop_runtime.panels import DesktopPanelComposer


class _PanelMission:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, *, mode: str = "json") -> dict:  # noqa: ARG002
        return dict(self._payload)


class _PanelRuntime:
    def __init__(self, *, latest_agent=None, missions=None) -> None:
        self._latest_agent = latest_agent
        self._missions = missions if missions is not None else []

    def desktop_agent_status(self):
        if self._latest_agent is None:
            return {"latest_mission": None}
        return {"latest_mission": self._latest_agent}

    def desktop_agent_list(self):
        return [_PanelMission(item) for item in self._missions]


class _PanelBridge:
    def __init__(self, *, latest_agent=None, missions=None, dashboard=None, health=None, timeline=None) -> None:
        self.runtime = _PanelRuntime(latest_agent=latest_agent, missions=missions)
        self._dashboard = dashboard if dashboard is not None else {}
        self._health = health if health is not None else {}
        self._timeline = timeline if timeline is not None else {}

    def hud_dashboard(self):
        return self._dashboard

    def hud_health(self):
        return self._health

    def hud_timeline(self, *, limit: int):  # noqa: ARG002
        return self._timeline


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


def test_panel_composer_handles_missing_agent_state() -> None:
    composer = DesktopPanelComposer(_PanelBridge(latest_agent=None, missions=[]))

    snapshot = composer.compose()

    assert snapshot.runtime_panels == []
    assert snapshot.missions == []


def test_panel_composer_handles_none_world_and_current_step() -> None:
    completed = {
        "mission_id": "mission-1",
        "goal": "crear carpeta",
        "status": "completed",
        "success": True,
        "world_state": {"current_step": None, "target_path": "C:/Users/GAMER/Documents/hola xd"},
        "completed_steps": ["create-folder"],
        "summary": "Mision completada",
    }
    composer = DesktopPanelComposer(_PanelBridge(latest_agent=completed, missions=[completed]))

    snapshot = composer.compose()

    assert snapshot.runtime_panels[0]["status"] == "completed"
    assert snapshot.runtime_panels[0]["current_step"] == "Sin paso activo"
    assert snapshot.missions[0].status == "completed"
    assert snapshot.missions[0].metadata["current_step"] == "Sin paso activo"
    assert snapshot.missions[0].metadata["completed_steps"] == ["create-folder"]


def test_panel_composer_handles_empty_current_step_shapes() -> None:
    for world_state in (None, {"current_step": {}}):
        mission = {
            "mission_id": "mission-shape",
            "goal": "crear carpeta",
            "status": "completed",
            "success": True,
            "world_state": world_state,
        }
        composer = DesktopPanelComposer(_PanelBridge(latest_agent=mission, missions=[mission]))

        snapshot = composer.compose()

        assert snapshot.runtime_panels[0]["current_step"] == "Sin paso activo"
        assert snapshot.missions[0].metadata["current_step"] == "Sin paso activo"


def test_panel_composer_degrades_to_safe_snapshot_on_bad_bridge(caplog) -> None:
    class _BrokenBridge(_PanelBridge):
        def hud_dashboard(self):
            raise RuntimeError("panel backend failed")

    composer = DesktopPanelComposer(_BrokenBridge())

    snapshot = composer.compose()

    assert snapshot.health_summary["aggregate_status"] == "degraded"
    assert snapshot.alerts[0]["title"] == "Desktop panel degraded"


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


def test_desktop_backend_start_does_not_auto_sync_indexing_by_default(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings, start=False)
    calls = 0
    original_run = app.indexing_runtime_service.run
    try:
        def _counted_run(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_run(*args, **kwargs)

        app.indexing_runtime_service.run = _counted_run  # type: ignore[method-assign]

        desktop.start_backend()

        assert calls == 0
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


def test_desktop_streaming_updates_single_assistant_message_and_speaks_at_done(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        gpt_oss_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    spoken: list[str] = []

    def _stream(_request, *, cancel_check=None):
        yield StreamChunk(text="Ho", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        assert spoken == []
        time.sleep(0.05)
        yield StreamChunk(text="la", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        yield StreamChunk(done=True, metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda text: spoken.append(text)  # type: ignore[method-assign]  # noqa: SLF001
    try:
        future = desktop.send_chat_async("hola jarvis", correlation_id="stream-1", metadata={"stream": True})
        deadline = time.perf_counter() + 5.0
        saw_partial = False
        while time.perf_counter() < deadline and not future.done():
            state = desktop.shell_state(force=True)
            assistant = [message for message in state.conversation if message.role == "assistant"]
            if assistant and assistant[-1].content == "Ho":
                saw_partial = True
            time.sleep(0.01)
        response = future.result(timeout=5)
        state = desktop.shell_state(force=True)
        assistant = [message for message in state.conversation if message.role == "assistant"]

        assert saw_partial
        assert assistant[-1].content == "Hola"
        assert response.raw_result["chunks"] == 2
        assert spoken == ["Hola"]
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_streaming_superseded_stream_does_not_mix_tokens(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        gpt_oss_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)

    def _stream(request, *, cancel_check=None):
        prompt = request.messages[-1].content
        if "primero" in prompt:
            yield StreamChunk(text="viejo", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
            time.sleep(0.2)
            yield StreamChunk(text="-tarde", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
            yield StreamChunk(done=True, metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
            return
        yield StreamChunk(text="nuevo", metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})
        yield StreamChunk(done=True, metadata={"provider": "gpt_oss", "model": "gpt-oss:20b"})

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        first = desktop.send_chat_async("primero", correlation_id="stream-old", metadata={"stream": True})
        time.sleep(0.05)
        second = desktop.send_chat_async("segundo", correlation_id="stream-new", metadata={"stream": True})
        first.result(timeout=5)
        second.result(timeout=5)
        contents = [message.content for message in desktop.shell_state(force=True).conversation if message.role == "assistant"]

        assert any(content == "nuevo" for content in contents)
        assert not any("viejo-tarde" in content for content in contents)
    finally:
        app.stop()
        desktop.shutdown()
