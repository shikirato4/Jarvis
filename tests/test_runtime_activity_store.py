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
