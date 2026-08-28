from __future__ import annotations

from jarvis.core.events import EventBus
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
    assert states["nemotron"] is RuntimeActivityState.SUCCESS
    assert states["action_executor"] is RuntimeActivityState.SUCCESS
    assert states["web"] is RuntimeActivityState.SUCCESS
    assert ("agent_controller", "nemotron") in edges
    assert ("nemotron", "agent_controller") in edges
    assert ("agent_controller", "action_executor") in edges
    assert ("action_executor", "web") in edges
    assert ("web", "agent_controller") in edges


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
