from __future__ import annotations

import time
from threading import Event

from jarvis.desktop import build_desktop_runtime
from jarvis.models_runtime.base import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
    StreamChunk,
)
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


def _install_fixed_agent_response(app, content: str = "Respuesta de Jarvis.") -> None:
    def infer(request):
        return ModelResponse(
            provider_name="nemotron-test",
            provider_kind=ProviderKind.LOCAL,
            logical_model=request.logical_model or "general_assistant",
            model_name="nemotron-test",
            content=content,
            latency_ms=1.0,
        )

    app.agent_controller._infer_model = infer  # noqa: SLF001


def test_desktop_runtime_boot_and_panels(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
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
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    captured: list[tuple[str, str | None]] = []
    requests = []
    try:
        def _scripted_infer(request):
            requests.append(request)
            if request.messages[-1].role == "tool":
                return ModelResponse(
                    provider_name="nemotron_omni_local",
                    provider_kind=ProviderKind.LOCAL,
                    logical_model=request.logical_model or "general_assistant",
                    model_name="nemotron-test",
                    content="La derivada es 2 x.",
                    latency_ms=1.0,
                )
            return ModelResponse(
                provider_name="nemotron_omni_local",
                provider_kind=ProviderKind.LOCAL,
                logical_model=request.logical_model or "general_assistant",
                model_name="nemotron-test",
                tool_calls=[
                    ModelToolCall(
                        id="science-tool-1",
                        name="science.solve_problem",
                        arguments={"query": "calcula la derivada de x**2"},
                    )
                ],
                latency_ms=1.0,
            )

        app.agent_controller._infer_model = _scripted_infer  # type: ignore[method-assign]  # noqa: SLF001
        original = app.voice_runtime_service.speak

        def _capture(text: str, *, correlation_id: str | None = None):
            captured.append((text, correlation_id))
            return original(text, correlation_id=correlation_id)

        app.voice_runtime_service.speak = _capture  # type: ignore[method-assign]
        response = desktop.send_chat("calcula la derivada de x**2")
        assert response.message.content == "La derivada es 2 x.", response.raw_result
        assert captured
        assert captured[-1][0] == response.spoken_content
        assert response.spoken_content != response.message.content
        assert "dos equis" in captured[-1][0].casefold()
        assert "metadata" not in captured[-1][0].casefold()
        assert len(requests) == 2
        assert requests[1].messages[-1].role == "tool"
        assert requests[1].messages[-1].name == "science.solve_problem"
    finally:
        app.stop()


def test_streaming_calculator_prompt_routes_through_nemotron_agent_tool_loop(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)
    requests = []
    first_final_chunk = Event()
    release_final_chunk = Event()
    try:
        def _scripted_stream(request, *, cancel_check=None):
            requests.append(request)
            assert cancel_check is not None
            if request.messages[-1].role == "tool":
                yield StreamChunk(
                    text="El resultado ",
                    metadata={"provider": "nemotron-test", "model": "nemotron-test"},
                )
                first_final_chunk.set()
                release_final_chunk.wait(timeout=10)
                yield StreamChunk(
                    text="es 73414.",
                    metadata={"provider": "nemotron-test", "model": "nemotron-test"},
                )
                yield StreamChunk(done=True, metadata={"finish_reason": "stop", "stream_closed_normally": True})
                return
            yield StreamChunk(
                tool_calls=[
                    ModelToolCall(
                        id="calculator-stream-1",
                        name="calculator.evaluate",
                        arguments={"expression": "781 * 94"},
                    )
                ],
                metadata={"provider": "nemotron-test", "model": "nemotron-test"},
            )
            yield StreamChunk(done=True, metadata={"finish_reason": "tool_calls", "stream_closed_normally": True})

        app.agent_controller._infer_model = lambda _request: (_ for _ in ()).throw(  # type: ignore[method-assign]  # noqa: SLF001
            AssertionError("agentic desktop streaming must use the streaming provider")
        )
        app.agent_controller._stream_model = _scripted_stream  # type: ignore[method-assign]  # noqa: SLF001
        app.runtime_service.stream_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("calculator chat must not use the parallel direct stream")
        )

        future = desktop.send_chat_async(
            "calcula 781 por 94",
            correlation_id="calculator-stream-agent",
            metadata={"stream": True},
        )
        assert first_final_chunk.wait(timeout=10)
        assert future.done() is False
        partial = desktop.shell_state(force=True).conversation[-1]
        assert partial.role == "assistant"
        assert partial.content == "El resultado "
        assert partial.metadata["streaming"] is True
        assert "Ejecutando Cálculo…" in partial.metadata["thinking"]["lines"]
        assert "Cálculo finalizó correctamente." in partial.metadata["thinking"]["lines"]
        release_final_chunk.set()
        response = future.result(timeout=10)

        assert response.message.content == "El resultado es 73414."
        assert len(requests) == 2
        assert requests[0].tools
        assert requests[1].messages[-2].tool_calls[0].name == "calculator.evaluate"
        assert requests[1].messages[-1].role == "tool"
        assert requests[1].messages[-1].name == "calculator.evaluate"
        assert response.raw_result["assistant_final_present"] is True
        assert response.raw_result["outcome_code"] == "NORMAL_STOP"
        assert response.raw_result["safe_reasoning_summary"]
        assert "reasoning_content" not in repr(response.raw_result)
        assert response.message.metadata["thought"]["summary"]
        recent_events = list(reversed(app.event_bus.recent_events()))
        event_names = [event["event_name"] for event in recent_events]
        context_index = event_names.index("conversation_context_built")
        expected_sequence = [
            "agent_controller.started",
            "agent_controller.model_requested",
            "agent_controller.tool_requested",
            "action_executor.started",
            "action_executor.succeeded",
            "agent_controller.tool_observed",
            "agent_controller.model_requested",
            "agent_controller.completed",
        ]
        cursor = context_index
        for expected in expected_sequence:
            cursor = event_names.index(expected, cursor + 1)
        assert "calcula 781 por 94" not in repr(recent_events)
        run_id = response.raw_result["run_id"]
        activity = app.runtime_activity_store.snapshot(run_id=run_id)
        edges = {(edge.source, edge.target) for edge in activity.edges}
        assert {
                ("model_provider", "agent_controller"),
            ("agent_controller", "action_executor"),
            ("action_executor", "calculator"),
            ("calculator", "action_executor"),
            ("action_executor", "agent_controller"),
                ("agent_controller", "model_provider"),
        } <= edges

        follow_up_requests = []

        def _follow_up_stream(request, **_kwargs):
            follow_up_requests.append(request)
            yield StreamChunk(
                text="Seguimos.",
                metadata={"provider": "nemotron-test", "model": "nemotron-test"},
            )
            yield StreamChunk(done=True, metadata={"finish_reason": "stop", "stream_closed_normally": True})

        app.runtime_service.stream_model = _follow_up_stream  # type: ignore[method-assign]
        desktop.send_chat_async(
            "explica el resultado anterior",
            correlation_id="calculator-follow-up",
            metadata={"stream": True},
        ).result(timeout=10)
        follow_up_messages = follow_up_requests[0].messages
        assert any(
            call.name == "calculator.evaluate"
            for message in follow_up_messages
            for call in message.tool_calls
        )
        assert any(
            message.role == "tool"
            and message.name == "calculator.evaluate"
            and '"value":73414' in message.content
            for message in follow_up_messages
        ), [
            (message.role, message.name, message.content, message.tool_call_id)
            for message in follow_up_messages
        ]
    finally:
        release_final_chunk.set()
        app.stop()


def test_desktop_streaming_context_survives_twenty_turns_tool_attachment_and_compaction(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_enabled=False,
        embeddings_enabled=False,
        # 1_700: the capability manifest now carries per-family Spanish
        # usage guides, so the composed system prompt legitimately grew.
        conversation_context_max_tokens=1_700,
        conversation_context_safety_margin_tokens=128,
    )
    app, desktop = build_desktop_runtime(settings)
    requests = []
    attachment = desktop.add_attachment_bytes(
        "contexto.txt",
        b"Referencia de adjunto: ALEACION-SEGURA.",
        "text/plain",
    )

    def _scripted_stream(request, **_kwargs):
        requests.append(request)
        if request.messages[-1].role == "tool":
            yield StreamChunk(text="El resultado es 73414.")
            yield StreamChunk(done=True, metadata={"finish_reason": "stop", "stream_closed_normally": True})
            return
        current = request.messages[-1].content.casefold()
        if "781" in current and "94" in current:
            yield StreamChunk(
                tool_calls=[
                    ModelToolCall(
                        id="context-calculator-1",
                        name="calculator.evaluate",
                        arguments={"expression": "781 * 94"},
                    )
                ]
            )
            yield StreamChunk(done=True, metadata={"finish_reason": "tool_calls", "stream_closed_normally": True})
            return
        if "cual era la palabra" in current or "cuál era la palabra" in current:
            yield StreamChunk(text="La palabra era COBALTO-721.")
            yield StreamChunk(done=True, metadata={"finish_reason": "stop", "stream_closed_normally": True})
            return
        yield StreamChunk(text="Entendido.")
        yield StreamChunk(done=True, metadata={"finish_reason": "stop", "stream_closed_normally": True})

    app.runtime_service.stream_model = _scripted_stream  # type: ignore[method-assign]
    app.agent_controller._stream_model = _scripted_stream  # type: ignore[method-assign]  # noqa: SLF001
    try:
        desktop.send_chat_async(
            "Durante esta conversación recuerda que la palabra es COBALTO-721.",
            correlation_id="context-opening",
            metadata={"stream": True, "attachment_ids": [attachment["id"]]},
        ).result(timeout=10)
        for index in range(10):
            desktop.send_chat_async(
                f"Turno previo {index}: " + ("contenido de relleno verificable " * 7),
                correlation_id=f"context-before-{index}",
                metadata={"stream": True},
            ).result(timeout=10)
        calculation = desktop.send_chat_async(
            "calcula 781 por 94",
            correlation_id="context-calculation",
            metadata={"stream": True},
        ).result(timeout=10)
        assert calculation.message.content == "El resultado es 73414."
        for index in range(10):
            desktop.send_chat_async(
                f"Turno posterior {index}: " + ("más contenido de relleno verificable " * 7),
                correlation_id=f"context-after-{index}",
                metadata={"stream": True},
            ).result(timeout=10)

        recalled = desktop.send_chat_async(
            "¿Cuál era la palabra?",
            correlation_id="context-recall",
            metadata={"stream": True},
        ).result(timeout=10)

        assert recalled.message.content == "La palabra era COBALTO-721."
        final_request = requests[-1]
        serialized = final_request.model_dump_json()
        assert "COBALTO-721" in serialized
        assert "ALEACION-SEGURA" in serialized
        assert any(
            call.name == "calculator.evaluate"
            for message in final_request.messages
            for call in message.tool_calls
        )
        assert any(
            message.role == "tool"
            and message.name == "calculator.evaluate"
            and '"value":73414' in message.content
            for message in final_request.messages
        )
        context = final_request.metadata["conversation_context"]
        assert context["history_message_count"] >= 44
        assert context["tool_observation_count"] >= 1
        assert context["attachment_context_count"] >= 1
        assert context["context_compacted"] is True
        assert context["context_truncated"] is False
        events = list(reversed(app.event_bus.recent_events()))
        recall_events = [
            event
            for event in events
            if event["payload"].get("correlation_id") == "context-recall"
        ]
        assert {event["event_name"] for event in recall_events} >= {
            "conversation_context_built",
            "context_compacted",
        }
        assert "COBALTO-721" not in repr(recall_events)
        assert "ALEACION-SEGURA" not in repr(recall_events)
    finally:
        app.stop()


def test_desktop_runtime_exposes_voice_controls(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
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
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_fixed_agent_response(app)
    original = desktop._chat.handle  # noqa: SLF001
    try:
        def _slow_handle(text: str, *args, **kwargs):
            time.sleep(1.0)
            return original(text, *args, **kwargs)

        desktop._chat.handle = _slow_handle  # type: ignore[method-assign]  # noqa: SLF001
        started = time.perf_counter()
        future = desktop.send_chat_async("calcula la derivada de x**2")
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5
        assert desktop.shell_state().busy is True
        response = future.result(timeout=15.0)
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
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    _install_fixed_agent_response(app)
    calls = 0
    original = desktop._chat.handle  # noqa: SLF001
    try:
        def _slow_handle(text: str, *args, **kwargs):
            nonlocal calls
            calls += 1
            time.sleep(0.2)
            return original(text, *args, **kwargs)

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
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    spoken: list[str] = []
    partial_chunk_yielded = Event()
    release_final_chunk = Event()

    def _stream(_request, *, cancel_check=None, execution_context=None):
        assert execution_context.request_id == "stream-1"
        partial_chunk_yielded.set()
        yield StreamChunk(text="**nave", metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"})
        assert spoken == []
        release_final_chunk.wait(timeout=10)
        yield StreamChunk(text="gación** ", metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"})
        yield StreamChunk(done=True, metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "finish_reason": "stop", "stream_closed_normally": True})

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda text: spoken.append(text)  # type: ignore[method-assign]  # noqa: SLF001
    try:
        future = desktop.send_chat_async("hola jarvis", correlation_id="stream-1", metadata={"stream": True})
        assert partial_chunk_yielded.wait(timeout=5)
        deadline = time.perf_counter() + 5.0
        saw_partial = False
        while time.perf_counter() < deadline and not future.done():
            state = desktop.shell_state(force=True)
            assistant = [message for message in state.conversation if message.role == "assistant"]
            if assistant and assistant[-1].content == "**nave":
                saw_partial = True
                release_final_chunk.set()
                break
            time.sleep(0.01)
        release_final_chunk.set()
        response = future.result(timeout=5)
        state = desktop.shell_state(force=True)
        assistant = [message for message in state.conversation if message.role == "assistant"]

        assert saw_partial
        authoritative = "**navegación** "
        assert assistant[-1].content == authoritative
        assert response.message.content == authoritative
        assert response.raw_result["content"] == authoritative
        assert response.raw_result["chunks"] == 2
        assert response.raw_result["provider_text_length"] == len(authoritative)
        assert response.raw_result["desktop_authoritative_text_length"] == len(authoritative)
        assert response.raw_result["final_render_source_length"] == len(authoritative)
        assert response.raw_result["final_render_completed"] is True
        assert response.raw_result["stream_closed_normally"] is True
        assert spoken == [authoritative]
    finally:
        release_final_chunk.set()
        app.stop()
        desktop.shutdown()


def test_desktop_length_finish_is_rendered_as_truncated_not_interrupted(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="local",
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)

    def _stream(_request, **_kwargs):
        yield StreamChunk(
            text="respuesta parcial",
            metadata={"provider": "qwen_local", "model": "qwen-test"},
        )
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "qwen_local",
                "model": "qwen-test",
                "finish_reason": "length",
                "stream_closed_normally": True,
                "incomplete": True,
                "completion_tokens": 500,
            },
        )

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    try:
        response = desktop.send_chat_async(
            "hola jarvis",
            correlation_id="stream-length",
            metadata={"stream": True},
        ).result(timeout=5)

        assert response.raw_result["status"] == "truncated"
        assert response.raw_result["outcome_code"] == "MODEL_OUTPUT_TRUNCATED"
        assert response.raw_result["finish_reason"] == "length"
        assert response.raw_result["completion_accepted"] is False
        assert response.raw_result["exception_category"] is None
        assert "respuesta parcial" in response.message.content
        assert "límite de salida" in response.message.content
        assert "interrumpió" not in response.message.content

        followup_requests: list[ModelRequest] = []

        def _followup_stream(request, **_kwargs):
            followup_requests.append(request)
            yield StreamChunk(
                text="continuación manual completa",
                metadata={"provider": "qwen_local", "model": "qwen-test"},
            )
            yield StreamChunk(
                done=True,
                metadata={
                    "provider": "qwen_local",
                    "model": "qwen-test",
                    "finish_reason": "stop",
                    "stream_closed_normally": True,
                    "incomplete": False,
                    "completion_tokens": 20,
                },
            )

        app.runtime_service.stream_model = _followup_stream  # type: ignore[method-assign]
        followup = desktop.send_chat_async(
            "continúa",
            correlation_id="manual-continue-after-length",
            metadata={"stream": True},
        ).result(timeout=5)

        assert followup.raw_result["status"] == "completed"
        assert followup_requests[0].metadata["manual_truncation_continue"] is True
        assert followup_requests[0].metadata["generation_profile"] == "detailed"
        assert any(
            message.role == "assistant"
            and "respuesta parcial" in message.content
            for message in followup_requests[0].messages
        )
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_auto_continuation_remains_one_logical_assistant_response(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="local",
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)
    requests: list[ModelRequest] = []

    def _coordinated_stream(request, **_kwargs):
        requests.append(request)
        if len(requests) == 1:
            yield StreamChunk(
                text="Primera parte y frase repetida",
                metadata={"provider": "qwen_local", "model": "qwen-test"},
            )
            yield StreamChunk(
                done=True,
                metadata={
                    "provider": "qwen_local",
                    "model": "qwen-test",
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "incomplete": True,
                    "completion_tokens": 100,
                },
            )
            return
        yield StreamChunk(
            text="frase repetida y final.",
            metadata={"provider": "qwen_local", "model": "qwen-test"},
        )
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "qwen_local",
                "model": "qwen-test",
                "finish_reason": "stop",
                "stream_closed_normally": True,
                "incomplete": False,
                "completion_tokens": 80,
            },
        )

    app.model_service._coordinator.stream = _coordinated_stream  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat_async(
            "Explica detalladamente un tema largo paso a paso.",
            correlation_id="desktop-auto-continuation",
            metadata={"stream": True},
        ).result(timeout=5)
        state = desktop.shell_state(force=True)
        assistant_messages = [
            message for message in state.conversation if message.role == "assistant"
        ]

        assert len(requests) == 2
        assert len(assistant_messages) == 1
        assert response.message.content == "Primera parte y frase repetida y final."
        assert response.raw_result["status"] == "completed"
        assert response.raw_result["auto_continuation_count"] == 1
        assert response.raw_result["segment_finish_reasons"] == ["length", "stop"]
        assert "Continuando respuesta…" in response.raw_result["thinking_lines"]
        assert assistant_messages[0].metadata["thought"]["status"] == "completed"
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_extended_math_continuation_is_one_completed_response(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        llm_mode="local",
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)
    requests: list[ModelRequest] = []

    def _coordinated_stream(request, **_kwargs):
        requests.append(request)
        if len(requests) == 1:
            yield StreamChunk(
                text="## Matemáticas\n\n$$\nG_{",
                metadata={"provider": "qwen_local", "model": "qwen-test"},
            )
            yield StreamChunk(
                done=True,
                metadata={
                    "provider": "qwen_local",
                    "model": "qwen-test",
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "incomplete": True,
                    "completion_tokens": 100,
                },
            )
            return
        yield StreamChunk(
            text="\\mu\\nu}=8\\pi T_{\\mu\\nu}\n$$\n\n## Conclusión\nCompleta.",
            metadata={"provider": "qwen_local", "model": "qwen-test"},
        )
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "qwen_local",
                "model": "qwen-test",
                "finish_reason": "stop",
                "stream_closed_normally": True,
                "incomplete": False,
                "completion_tokens": 80,
            },
        )

    app.model_service._coordinator.stream = _coordinated_stream  # type: ignore[method-assign]  # noqa: SLF001
    try:
        response = desktop.send_chat_async(
            "explicame muy detaslladamente y muy claramente, muy muy a fondo "
            "con formulas y matematicas la relatividad general y la historia",
            correlation_id="desktop-extended-continuation",
            metadata={"stream": True},
        ).result(timeout=5)
        state = desktop.shell_state(force=True)
        assistant_messages = [
            message for message in state.conversation if message.role == "assistant"
        ]

        assert len(requests) == 2
        assert requests[0].max_tokens == 4096
        assert requests[0].metadata["output_budget"]["request_class"] == (
            "EXTENDED_LONG_FORM"
        )
        assert requests[1].max_tokens == 2500
        assert requests[1].metadata["reasoning_preference"] == "off"
        assert len(assistant_messages) == 1
        assert response.raw_result["status"] == "completed"
        assert response.raw_result["finish_reason"] == "stop"
        assert response.raw_result["segment_max_tokens"] == [4096, 2500]
        assert response.raw_result["output_request_class"] == "EXTENDED_LONG_FORM"
        assert "límite de salida" not in response.message.content
        assert "$$" in response.message.content
        assert "Continuando respuesta…" in response.raw_result["thinking_lines"]
        assert assistant_messages[0].metadata["thought"]["status"] == "completed"
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_unexpected_stream_eof_is_interrupted(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)

    def _stream(_request, **_kwargs):
        yield StreamChunk(
            text="prefijo íntegro",
            metadata={"provider": "qwen_local", "model": "qwen-test"},
        )
        # EOF without the provider's terminal event.

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    try:
        response = desktop.send_chat_async(
            "hola jarvis",
            correlation_id="stream-eof",
            metadata={"stream": True},
        ).result(timeout=5)

        assert response.raw_result["status"] == "interrupted"
        assert response.raw_result["finish_reason"] is None
        assert response.raw_result["stream_closed_normally"] is False
        assert response.raw_result["completion_accepted"] is False
        assert response.raw_result["exception_category"] == "STREAM_TRANSPORT"
        assert "prefijo íntegro" in response.message.content
        assert "interrumpió" in response.message.content
    finally:
        app.stop()
        desktop.shutdown()


def test_dns_action_routes_to_agent_but_informational_control_does_not(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        assert desktop._should_route_stream_to_agent("limpia mi cache DNS") is True  # noqa: SLF001
        assert (  # noqa: SLF001
            desktop._should_route_stream_to_agent(  # noqa: SLF001
                "¿Cómo puedo limpiar mi cache DNS?"
            )
            is False
        )
        assert (  # noqa: SLF001
            desktop._should_route_stream_to_agent(  # noqa: SLF001
                "busca en mis archivos la ruta de Hogwarts Legacy y dame la direccion"
            )
            is True
        )
        assert (  # noqa: SLF001
            desktop._should_route_stream_to_agent(  # noqa: SLF001
                "Escribe una explicación detallada de cómo hacer composta doméstica."
            )
            is False
        )
    finally:
        if app.started:
            app.stop()
        desktop.shutdown()


def test_normal_chat_thinking_is_conservative_and_thought_uses_safe_summary(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)

    def _stream(_request, **_kwargs):
        yield StreamChunk(
            metadata={
                "provider": "qwen_local",
                "model": "Qwen3.5-9B-IQ4_XS",
                "reasoning_active": True,
            }
        )
        yield StreamChunk(text="Respuesta completa.")
        yield StreamChunk(
            done=True,
            metadata={
                "provider": "qwen_local",
                "model": "Qwen3.5-9B-IQ4_XS",
                "finish_reason": "stop",
                "stream_closed_normally": True,
            },
        )

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    try:
        response = desktop.send_chat_async(
            "Explica una idea sencilla",
            correlation_id="normal-safe-thought",
            metadata={"stream": True},
        ).result(timeout=10)

        assert response.raw_result["status"] == "completed"
        assert response.raw_result["outcome_code"] == "NORMAL_STOP"
        assert "contenido interno se mantiene privado" not in response.message.metadata[
            "thought"
        ]["summary"]
        assert response.message.metadata["thought"]["elapsed_seconds"] >= 0
        lines = response.message.metadata["thinking"]["lines"]
        assert lines == ["Analizando la solicitud…", "Preparando respuesta…"]
        assert not any(
            marker in " ".join(lines)
            for marker in ("Terminal", "Archivos", "herramienta")
        )
        assert "reasoning_content" not in repr(response.message.metadata)
    finally:
        app.stop()
        desktop.shutdown()


def test_desktop_streaming_superseded_stream_does_not_mix_tokens(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)

    def _stream(request, *, cancel_check=None, execution_context=None):
        assert execution_context.request_id in {"stream-old", "stream-new"}
        prompt = request.messages[-1].content
        if "primero" in prompt:
            yield StreamChunk(text="viejo", metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"})
            time.sleep(0.2)
            yield StreamChunk(text="-tarde", metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"})
            yield StreamChunk(done=True, metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "finish_reason": "stop", "stream_closed_normally": True})
            return
        yield StreamChunk(text="nuevo", metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"})
        yield StreamChunk(done=True, metadata={"provider": "nemotron_omni_local", "model": "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4", "finish_reason": "stop", "stream_closed_normally": True})

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


def test_superseding_stream_cancels_at_admission_when_workers_are_saturated(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        voice_audio_output_backend_default="in_memory",
        voice_tts_provider_default="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    release_workers = Event()
    worker_a_started = Event()
    worker_b_started = Event()
    old_chunk_started = Event()
    release_old_chunk = Event()

    def _occupy_worker(started: Event) -> None:
        started.set()
        release_workers.wait(timeout=10)

    blocker_a = desktop._executor.submit(_occupy_worker, worker_a_started)  # noqa: SLF001
    blocker_b = desktop._executor.submit(_occupy_worker, worker_b_started)  # noqa: SLF001

    def _stream(request, *, cancel_check=None, execution_context=None):
        prompt = request.messages[-1].content
        if "primero" in prompt:
            old_chunk_started.set()
            yield StreamChunk(text="viejo", metadata={"provider": "nemotron-test", "model": "nemotron-test"})
            release_old_chunk.wait(timeout=10)
            yield StreamChunk(text="-tarde", metadata={"provider": "nemotron-test", "model": "nemotron-test"})
            yield StreamChunk(done=True, metadata={"provider": "nemotron-test", "model": "nemotron-test", "finish_reason": "stop", "stream_closed_normally": True})
            return
        yield StreamChunk(text="nuevo", metadata={"provider": "nemotron-test", "model": "nemotron-test"})
        yield StreamChunk(done=True, metadata={"provider": "nemotron-test", "model": "nemotron-test", "finish_reason": "stop", "stream_closed_normally": True})

    app.runtime_service.stream_model = _stream  # type: ignore[method-assign]
    desktop._voice.speak_response = lambda _text: None  # type: ignore[method-assign]  # noqa: SLF001
    try:
        assert worker_a_started.wait(timeout=5)
        assert worker_b_started.wait(timeout=5)
        first = desktop.send_chat_async("primero", correlation_id="saturated-old", metadata={"stream": True})
        assert old_chunk_started.wait(timeout=5)

        second = desktop.send_chat_async("segundo", correlation_id="saturated-new", metadata={"stream": True})
        release_old_chunk.set()
        first.result(timeout=5)
        release_workers.set()
        second.result(timeout=5)

        contents = [
            message.content
            for message in desktop.shell_state(force=True).conversation
            if message.role == "assistant"
        ]
        assert any(content == "nuevo" for content in contents)
        assert not any("viejo-tarde" in content for content in contents)
    finally:
        release_old_chunk.set()
        release_workers.set()
        blocker_a.result(timeout=5)
        blocker_b.result(timeout=5)
        app.stop()
        desktop.shutdown()
