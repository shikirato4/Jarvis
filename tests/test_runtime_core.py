import pytest

from jarvis.core.errors import SafetyViolationError
from jarvis.routing.models import TaskRequest


def test_runtime_snapshot_contains_service_and_tool_catalog(jarvis_app) -> None:
    snapshot = jarvis_app.runtime_service.snapshot()
    service_names = {service.name for service in snapshot.services}
    assert {"jarvis.app", "memory", "modules", "automation", "task_router", "runtime"} <= service_names
    assert "memory.lookup" in snapshot.tool_names
    assert "memory.store" in snapshot.action_names


def test_metacommand_mode_switch_updates_state(jarvis_app) -> None:
    response = jarvis_app.runtime_service.route({"raw_input": "/mode research switching"})
    assert response.mode == "research"
    assert response.state_snapshot is not None
    assert response.state_snapshot.mode.active_mode == "research"


def test_tool_invocation_routes_through_action_layer(jarvis_app) -> None:
    jarvis_app.action_router.execute(
        "memory.store",
        {"kind": "note", "content": "tool lookup verification", "source": "test"},
    )
    receipt = jarvis_app.runtime_service.invoke_tool("memory.lookup", {"query": "lookup", "limit": 5})
    assert receipt.status.value == "success"
    assert receipt.data["count"] >= 1


def test_operator_tool_requires_operator_mode(jarvis_app) -> None:
    with pytest.raises(SafetyViolationError):
        jarvis_app.runtime_service.route(
            TaskRequest(
                raw_input='/tool shell.command {"command":["where","python"],"timeout_seconds":5}',
                dry_run=True,
            )
        )

    jarvis_app.runtime_service.switch_mode("operator", reason="test")
    response = jarvis_app.runtime_service.route(
        TaskRequest(
            raw_input='/tool shell.command {"command":["where","python"],"timeout_seconds":5}',
            dry_run=True,
        )
    )
    assert response.tool_receipt is not None
    assert response.tool_receipt.message == "command validated in dry-run mode"


def test_state_metacommand_returns_runtime_snapshot(jarvis_app) -> None:
    response = jarvis_app.runtime_service.route({"raw_input": "/state"})
    assert response.state_snapshot is not None
    assert response.state_snapshot.mode.active_mode == jarvis_app.mode_manager.current_mode().value


def test_keyword_intent_inference_avoids_partial_token_misclassification(jarvis_app) -> None:
    inferred = jarvis_app.capability_registry.infer_intent("Necesito un brief sobre runtime", default_intent="research")
    assert inferred == "research_brief"
