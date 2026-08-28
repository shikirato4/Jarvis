from __future__ import annotations

from jarvis.desktop_runtime.activity_graph import build_activity_graph_scene
from jarvis.runtime_activity import (
    RuntimeActivityEdge,
    RuntimeActivityNode,
    RuntimeActivitySnapshot,
    RuntimeActivityState,
)


def test_activity_graph_contains_only_nodes_and_flows_present_in_snapshot() -> None:
    snapshot = RuntimeActivitySnapshot(
        nodes=[
            RuntimeActivityNode(
                component="nemotron",
                state=RuntimeActivityState.SUCCESS,
                run_id="graph-run",
            ),
            RuntimeActivityNode(
                component="agent_controller",
                state=RuntimeActivityState.ACTIVE,
                run_id="graph-run",
            ),
            RuntimeActivityNode(
                component="web",
                state=RuntimeActivityState.SUCCESS,
                run_id="graph-run",
            ),
        ],
        edges=[
            RuntimeActivityEdge(
                source="nemotron",
                target="agent_controller",
                run_id="graph-run",
            ),
            RuntimeActivityEdge(
                source="action_executor",
                target="web",
                run_id="graph-run",
            ),
        ],
    )

    scene = build_activity_graph_scene(snapshot, width=800, height=500)

    assert {node.component for node in scene.nodes} == {
        "nemotron",
        "agent_controller",
        "web",
    }
    assert [(edge.source, edge.target) for edge in scene.edges] == [
        ("nemotron", "agent_controller")
    ]
    assert "neural_router" not in repr(scene)


def test_empty_runtime_activity_renders_an_empty_graph() -> None:
    scene = build_activity_graph_scene(
        RuntimeActivitySnapshot(),
        width=800,
        height=500,
    )

    assert scene.nodes == []
    assert scene.edges == []
