from __future__ import annotations

from pathlib import Path

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
