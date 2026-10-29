from __future__ import annotations

from jarvis.core.events import EventBus
from jarvis.desktop_runtime.progress import project_runtime_progress
from jarvis.runtime_activity import RuntimeActivityState, RuntimeActivityStore


def test_activity_store_is_empty_until_real_runtime_events_arrive() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)

    snapshot = store.snapshot()

    assert snapshot.nodes == []
    assert snapshot.edges == []
    assert snapshot.events == []


def test_real_agent_tool_flow_builds_nodes_and_edges_without_fake_activity() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "run-activity", "correlation_id": "corr-activity"}

    events.publish("agent_controller.started", {**common, "step_count": 0})
    events.publish("agent_controller.model_requested", {**common, "step_count": 0})
    events.publish(
        "agent_controller.tool_requested",
        {
            **common,
            "step_count": 0,
            "call_id": "call-web",
            "action": "web.search",
        },
    )
    events.publish(
        "action_executor.started",
        {
            **common,
            "call_id": "call-web",
            "action": "web.search",
            "status": "active",
        },
    )
    events.publish(
        "action_executor.succeeded",
        {
            **common,
            "call_id": "call-web",
            "action": "web.search",
            "status": "success",
            "latency_ms": 12.5,
        },
    )
    events.publish(
        "agent_controller.tool_observed",
        {
            **common,
            "step_count": 1,
            "call_id": "call-web",
            "action": "web.search",
            "status": "success",
        },
    )
    events.publish("agent_controller.completed", {**common, "step_count": 1})

    snapshot = store.snapshot(run_id="run-activity")
    states = {node.component: node.state for node in snapshot.nodes}
    edges = {(edge.source, edge.target) for edge in snapshot.edges}

    assert states["agent_controller"] is RuntimeActivityState.SUCCESS
    assert states["model_provider"] is RuntimeActivityState.SUCCESS
    assert states["action_executor"] is RuntimeActivityState.SUCCESS
    assert states["web"] is RuntimeActivityState.SUCCESS
    assert ("agent_controller", "model_provider") in edges
    assert ("model_provider", "agent_controller") in edges
    assert ("agent_controller", "action_executor") in edges
    assert ("action_executor", "web") in edges
    assert ("web", "action_executor") in edges
    assert ("action_executor", "agent_controller") in edges


def test_activity_store_drops_private_payload_values_and_fake_progress() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    secret = "PRIVATE-PROMPT-CONTENT-511"

    events.publish(
        "image.generation_progress",
        {
            "run_id": "image-run",
            "prompt": secret,
            "status": "generating",
            "percent": 68,
        },
    )
    indeterminate = store.snapshot(run_id="image-run")
    events.publish(
        "image.generation_progress",
        {
            "run_id": "image-run",
            "prompt": secret,
            "status": "generating",
            "step": 20,
            "total_steps": 30,
        },
    )
    measurable = store.snapshot(run_id="image-run")

    assert secret not in repr(indeterminate)
    assert indeterminate.nodes[0].progress is None
    assert measurable.nodes[0].progress == 20 / 30
    assert measurable.nodes[0].step == 20
    assert measurable.nodes[0].total_steps == 30


def test_thinking_projection_preserves_real_multi_tool_order_without_duplicates() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "run-order", "correlation_id": "corr-order"}
    sequence = [
        ("agent_controller.started", {}),
        ("agent_controller.model_requested", {}),
        (
            "agent_controller.tool_requested",
            {"call_id": "files-1", "action": "files.search"},
        ),
        (
            "action_executor.started",
            {"call_id": "files-1", "action": "files.search", "status": "active"},
        ),
        (
            "action_executor.started",
            {"call_id": "files-1", "action": "files.search", "status": "active"},
        ),
        (
            "action_executor.succeeded",
            {"call_id": "files-1", "action": "files.search", "status": "success"},
        ),
        (
            "agent_controller.tool_requested",
            {"call_id": "terminal-1", "action": "terminal.run"},
        ),
        (
            "action_executor.started",
            {"call_id": "terminal-1", "action": "terminal.run", "status": "active"},
        ),
        (
            "action_executor.failed",
            {"call_id": "terminal-1", "action": "terminal.run", "status": "failed"},
        ),
        (
            "agent_controller.failed",
            {"outcome_code": "TOOL_EXECUTION_ERROR"},
        ),
    ]
    for event_name, extra in sequence:
        events.publish(event_name, {**common, **extra})

    projection = project_runtime_progress(
        store.snapshot(correlation_id="corr-order")
    )

    assert projection.status == "failed"
    assert projection.lines.count("Ejecutando Archivos…") == 1
    assert projection.lines.index("Ejecutando Archivos…") < projection.lines.index(
        "Archivos finalizó correctamente."
    )
    assert projection.lines.index("Ejecutando Terminal…") < projection.lines.index(
        "Terminal falló."
    )
    assert "command" not in repr(projection)


def test_thinking_projection_surfaces_confirmation_and_security_denial() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "run-security", "correlation_id": "corr-security"}
    events.publish(
        "action_executor.waiting_confirmation",
        {**common, "call_id": "one", "action": "terminal.run"},
    )
    waiting = project_runtime_progress(store.snapshot(correlation_id="corr-security"))
    assert waiting.status == "waiting"
    assert "La acción requiere confirmación." in waiting.lines

    events.publish(
        "action_executor.blocked",
        {**common, "call_id": "two", "action": "terminal.run"},
    )
    blocked = project_runtime_progress(store.snapshot(correlation_id="corr-security"))
    assert blocked.status == "blocked"
    assert "SecurityGate bloqueó la acción." in blocked.lines


def test_thinking_projection_surfaces_real_textual_continuation_event() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    events.publish(
        "model.auto_continuation_started",
        {
            "run_id": "continuation-run",
            "correlation_id": "continuation-correlation",
            "attempt": 1,
        },
    )

    projection = project_runtime_progress(
        store.snapshot(correlation_id="continuation-correlation")
    )

    assert projection.lines == ("Continuando respuesta…",)
    assert "continuación textual acotada" in str(projection.safe_summary)


def test_terminal_events_project_safe_lifecycle_without_output_text() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "terminal-run", "correlation_id": "terminal-corr", "execution_id": "exec-1"}

    events.publish("terminal.session_started", {**common, "session_id": "session-1", "pid": 42})
    events.publish("terminal.command_started", {**common, "session_id": "session-1", "status": "running"})
    events.publish(
        "terminal.output",
        {**common, "stream": "stdout", "sequence": 1, "characters": 18, "text": "SECRET-COMMAND-OUTPUT"},
    )
    events.publish("terminal.command_completed", {**common, "status": "succeeded", "exit_code": 0})
    events.publish("terminal.session_closed", {**common, "session_id": "session-1", "pid": 42})

    snapshot = store.snapshot(run_id="terminal-run")
    assert [event.event_name for event in snapshot.events] == [
        "terminal.command_started",
        "terminal.output",
        "terminal.command_completed",
    ]
    assert snapshot.nodes[0].component == "terminal"
    assert snapshot.nodes[0].state is RuntimeActivityState.SUCCESS
    assert all(event.action == "terminal.run" for event in snapshot.events)
    assert all(event.call_id == "exec-1" for event in snapshot.events)
    assert "SECRET-COMMAND-OUTPUT" not in repr(snapshot)


def test_terminal_events_without_run_id_are_not_assigned_to_agent_activity() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    events.publish(
        "terminal.command_started",
        {"correlation_id": "standalone", "execution_id": "exec-standalone", "status": "running"},
    )
    assert store.snapshot().events == []


def test_terminal_failure_timeout_and_cancel_states_are_distinct_events() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "terminal-states", "correlation_id": "corr", "execution_id": "exec"}
    events.publish("terminal.command_failed", {**common, "error_code": "PROCESS_FAILED"})
    events.publish("terminal.command_timed_out", {**common, "error_code": "TIMED_OUT"})
    events.publish("terminal.command_cancelled", {**common, "error_code": "CANCELLED"})
    snapshot = store.snapshot(run_id="terminal-states")
    assert [event.event_name for event in snapshot.events] == [
        "terminal.command_failed",
        "terminal.command_timed_out",
        "terminal.command_cancelled",
    ]
    assert [event.error_type for event in snapshot.events] == [
        "PROCESS_FAILED", "TIMED_OUT", "CANCELLED",
    ]
    assert snapshot.nodes[0].state is RuntimeActivityState.CANCELLED


def test_terminal_progress_is_concise_redacted_and_deduplicated() -> None:
    events = EventBus()
    store = RuntimeActivityStore(events)
    common = {"run_id": "terminal-progress", "correlation_id": "corr", "execution_id": "exec"}
    events.publish("terminal.command_started", common)
    events.publish("terminal.command_started", common)
    for sequence in range(1, 4):
        events.publish(
            "terminal.output",
            {**common, "sequence": sequence, "characters": 12, "text": "secret stdout"},
        )
    projection = project_runtime_progress(store.snapshot(correlation_id="corr"))
    assert projection.status == "active"
    assert projection.lines == ("Ejecutando Terminal…",)
    assert "secret stdout" not in repr(projection)


def test_terminal_progress_distinguishes_timeout_from_failure_and_cancel() -> None:
    for event_name, expected_status, expected_line in (
        ("terminal.command_failed", "failed", "Terminal falló."),
        ("terminal.command_timed_out", "failed", "Terminal agotó el tiempo."),
        ("terminal.command_cancelled", "cancelled", "Terminal fue cancelada."),
    ):
        events = EventBus()
        store = RuntimeActivityStore(events)
        events.publish(
            event_name,
            {"run_id": event_name, "correlation_id": "corr", "execution_id": "exec"},
        )
        projection = project_runtime_progress(store.snapshot(correlation_id="corr"))
        assert projection.status == expected_status
        assert projection.lines == (expected_line,)
