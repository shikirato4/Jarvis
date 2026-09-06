from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows terminal runtime")


def test_registered_terminal_tool_uses_secured_action_executor(jarvis_app, tmp_path) -> None:
    names = {definition.name for definition in jarvis_app.tool_registry.list_tools()}
    assert {"terminal.start_session", "terminal.run", "terminal.read_output", "terminal.cancel"} <= names

    started = jarvis_app.tool_registry.invoke(
        "terminal.start_session",
        {"shell": "powershell", "cwd": str(tmp_path)},
        correlation_id="terminal-app-e2e",
    )
    session_id = started.data["session_id"]
    executed = jarvis_app.tool_registry.invoke(
        "terminal.run",
        {
            "session_id": session_id,
            "command": "Write-Output secured-terminal-path",
            "timeout_seconds": 5,
        },
        correlation_id="terminal-app-e2e-run",
    )

    assert executed.status.value == "success"
    assert executed.data["exit_code"] == 0
    assert "secured-terminal-path" in executed.data["stdout"]
    events = jarvis_app.event_bus.recent_events()
    assert any(
        item["event_name"] == "action_executor.succeeded"
        and item["payload"]["action"] == "terminal.run"
        for item in events
    )
    jarvis_app.terminal_runtime.close_session(session_id)


def test_dangerous_terminal_tool_is_blocked_before_runtime(jarvis_app, tmp_path) -> None:
    started = jarvis_app.tool_registry.invoke(
        "terminal.start_session",
        {"shell": "cmd", "cwd": str(tmp_path)},
    )
    receipt = jarvis_app.tool_registry.invoke(
        "terminal.run",
        {"session_id": started.data["session_id"], "command": "shutdown /s"},
    )
    assert receipt.status.value == "failed"
    assert receipt.message == "Action blocked by the security gate."
    assert receipt.data["error_code"] == "SECURITY_BLOCKED"
    jarvis_app.terminal_runtime.close_session(started.data["session_id"])
