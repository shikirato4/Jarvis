from pathlib import Path

import pytest

from jarvis.actions.models import ActionStep
from jarvis.core.errors import ActionNotFoundError


def test_bootstrap_registers_modules_and_actions(jarvis_app) -> None:
    description = jarvis_app.describe()
    action_names = {item["name"] for item in description["actions"]}
    tool_names = {item["name"] for item in description["tools"]}
    assert "memory.store" in action_names
    assert "research.workspace_search" in action_names
    assert "writer.compose_note" in action_names
    assert "memory.lookup" in tool_names
    assert "runtime.snapshot" in tool_names


def test_memory_store_and_search_roundtrip(jarvis_app) -> None:
    jarvis_app.action_router.execute(
        "memory.store",
        {"kind": "fact", "content": "Jarvis puede recordar tareas", "source": "test"},
    )
    receipt = jarvis_app.action_router.execute(
        "memory.search",
        {"query": "recordar", "limit": 5},
    )
    assert receipt.data["count"] >= 1


def test_writer_rollback_removes_artifact_and_memory(jarvis_app, tmp_path: Path) -> None:
    output_file = tmp_path / "brief.md"
    steps = [
        ActionStep(
            action="writer.compose_note",
            payload={
                "title": "Rollback brief",
                "objective": "Validar rollback",
                "findings": ["Primer hallazgo"],
                "output_path": str(output_file),
                "persist_memory": True,
            },
        ),
        ActionStep(action="missing.action", payload={}),
    ]

    with pytest.raises(ActionNotFoundError):
        jarvis_app.action_router.execute_plan(steps)

    assert not output_file.exists()
    matches = jarvis_app.memory_service.search_memories("Rollback brief", limit=10)
    assert matches == []


def test_automation_can_be_persisted(jarvis_app) -> None:
    entry = jarvis_app.automation_service.save(
        {
            "name": "remember-heartbeat",
            "action_name": "memory.store",
            "payload": {"kind": "heartbeat", "content": "automation ok", "source": "automation"},
            "interval_seconds": 60,
            "enabled": True,
        }
    )
    assert entry.name == "remember-heartbeat"
    assert len(jarvis_app.automation_service.list()) == 1
