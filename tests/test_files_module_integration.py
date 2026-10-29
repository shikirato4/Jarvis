from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.action_executor import ActionRequest, ActionStatus
from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _application(tmp_path: Path):
    return build_application(
        Settings(
            workspace_root=tmp_path,
            data_dir=tmp_path / "runtime",
            ui_backend_kind="in_memory",
            system_backend_kind="in_memory",
        )
    )


def test_files_registered_in_actions_tools_and_capabilities(tmp_path: Path) -> None:
    app = _application(tmp_path)
    action_names = {item.name for item in app.action_registry.list_actions()}
    tool_names = {item.name for item in app.tool_registry.list_tools()}
    assert {"files.list", "files.search", "files.create_file", "files.move", "files.delete"} <= action_names
    assert {"files.list", "files.search", "files.create_file", "files.move", "files.delete"} <= tool_names
    descriptor = next(item.descriptor for item in app.capability_registry.list_capabilities() if item.descriptor.name == "files.navigator")
    assert "files.search" in descriptor.tool_names
    app.stop()


def test_safe_create_flows_through_action_executor_and_activity(tmp_path: Path) -> None:
    app = _application(tmp_path)
    request = ActionRequest(run_id="files-run", call_id="create-1", action="files.create_file", arguments={"path": "jarvis_test.txt", "content": "FILES_READY"})
    observation = app.action_executor.execute(request)
    assert observation.status is ActionStatus.SUCCESS
    assert observation.data["verified"] is True
    assert (tmp_path / "jarvis_test.txt").read_text(encoding="utf-8") == "FILES_READY"
    snapshot = app.runtime_activity_store.snapshot(run_id="files-run")
    assert any(node.component == "files" for node in snapshot.nodes)
    assert any(node.component == "file_navigator" for node in snapshot.nodes)
    app.stop()


def test_move_delete_and_replace_require_confirmation(tmp_path: Path) -> None:
    app = _application(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("x", encoding="utf-8")
    move = app.action_executor.execute(ActionRequest(run_id="r", call_id="m", action="files.move", arguments={"source": str(source), "destination": str(tmp_path / "moved.txt")}))
    assert move.status is ActionStatus.CONFIRMATION_REQUIRED
    replace = app.action_executor.execute(ActionRequest(run_id="r", call_id="c", action="files.create_file", arguments={"path": str(source), "content": "new", "overwrite": "replace"}))
    assert replace.status is ActionStatus.CONFIRMATION_REQUIRED
    delete = app.action_executor.execute(ActionRequest(run_id="r", call_id="d", action="files.delete", arguments={"path": str(source), "permanent": True}))
    assert delete.status is ActionStatus.CONFIRMATION_REQUIRED
    assert source.exists()
    app.stop()


def test_tool_invocation_uses_secured_executor(tmp_path: Path) -> None:
    app = _application(tmp_path)
    receipt = app.tool_registry.invoke("files.create_directory", {"path": "tools-dir"}, correlation_id="tool-files")
    assert receipt.status.value == "success"
    assert receipt.data["verified"] is True
    assert (tmp_path / "tools-dir").is_dir()
    app.stop()


@pytest.mark.parametrize("operation", ["read_text", "stat", "copy", "move"])
def test_isolated_multistep_files_trajectory(tmp_path: Path, operation: str) -> None:
    app = _application(tmp_path)
    source = tmp_path / "trajectory-fixture.txt"
    source.write_text("observed fixture content", encoding="utf-8")
    search = app.action_executor.execute(ActionRequest(
        run_id="trajectory", call_id="search", action="files.search",
        arguments={"pattern": source.name, "scope": str(tmp_path)},
    ))
    assert search.status is ActionStatus.SUCCESS
    observed_path = next(
        item["path"] for item in search.data["items"]
        if Path(item["path"]).name == source.name
    )
    assert Path(observed_path).resolve() == source.resolve()

    destination = tmp_path / f"{operation}-verified.txt"
    action = f"files.{operation}"
    arguments = (
        {"source": observed_path, "destination": str(destination)}
        if operation in {"copy", "move"} else {"path": observed_path}
    )
    request = ActionRequest(
        run_id="trajectory", call_id=operation, action=action, arguments=arguments,
    )
    observation = app.action_executor.execute(request)
    if operation in {"copy", "move"}:
        assert observation.status is ActionStatus.CONFIRMATION_REQUIRED
        assert app.action_executor.approve_confirmation(observation.confirmation_id)
        observation = app.action_executor.execute(request.model_copy(update={
            "confirmation_token": observation.confirmation_id,
        }))
    assert observation.status is ActionStatus.SUCCESS
    if operation == "read_text":
        assert "observed fixture content" in str(observation.data)
    elif operation == "stat":
        assert observation.data["metadata"]["size"] == source.stat().st_size
    else:
        assert destination.read_text(encoding="utf-8") == "observed fixture content"
        assert source.exists() is (operation == "copy")
    app.stop()


def test_file_observations_do_not_turn_missing_or_invalid_targets_into_success(
    tmp_path: Path,
) -> None:
    app = _application(tmp_path)
    missing = app.action_executor.execute(ActionRequest(
        run_id="missing", call_id="search", action="files.search",
        arguments={"pattern": "not-present-*.txt", "scope": str(tmp_path)},
    ))
    assert missing.status is ActionStatus.SUCCESS
    assert missing.data["items"] == []

    invalid = app.action_executor.execute(ActionRequest(
        run_id="invalid", call_id="read", action="files.read_text",
        arguments={"path": 42},
    ))
    assert invalid.status is ActionStatus.FAILED
    assert invalid.error_type == "ActionValidationError"

    stale = app.action_executor.execute(ActionRequest(
        run_id="stale", call_id="read", action="files.read_text",
        arguments={"path": str(tmp_path / "disappeared.txt")},
    ))
    assert stale.status is ActionStatus.FAILED
    app.stop()
