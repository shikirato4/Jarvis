from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from jarvis.action_executor import ActionRequest, SecurityDecisionStatus, TypedActionSecurityGate
from jarvis.actions.models import ActionRisk
from jarvis.actions.registry import ActionRegistry
from jarvis.modules.terminal_module import RunTerminalPayload, StartSessionPayload, TerminalModule
from jarvis.terminal_runtime import (
    ExecutionState,
    TerminalError,
    TerminalErrorCode,
    TerminalRuntime,
    TerminalSessionState,
)


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows terminal runtime")


@pytest.fixture
def runtime():
    value = TerminalRuntime(max_output_chars=4096)
    yield value
    value.shutdown()


def test_create_powershell_session(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    assert session.shell == "powershell"
    assert session.state is TerminalSessionState.READY
    assert session.owned_by_jarvis and session.pid


def test_terminal_events_propagate_agent_context(tmp_path):
    events: list[tuple[str, dict]] = []
    runtime = TerminalRuntime(event_sink=lambda name, payload: events.append((name, payload)))
    try:
        session = runtime.start_session(
            "powershell",
            cwd=tmp_path,
            run_id="agent-run",
            correlation_id="agent-correlation",
        )
        result = runtime.run(
            session.session_id,
            "Write-Output terminal-context",
            run_id="agent-run",
            correlation_id="agent-correlation",
        )
        started = next(payload for name, payload in events if name == "terminal.command_started")
        completed = next(payload for name, payload in events if name == "terminal.command_completed")
        output = next(payload for name, payload in events if name == "terminal.output")
        assert result.status is ExecutionState.SUCCEEDED
        for payload in (started, completed, output):
            assert payload["run_id"] == "agent-run"
            assert payload["correlation_id"] == "agent-correlation"
        assert "text" not in output
        assert "command" not in completed
    finally:
        runtime.shutdown()


def test_create_cmd_session(runtime, tmp_path):
    session = runtime.start_session("cmd", cwd=tmp_path)
    assert session.shell == "cmd"
    assert session.state is TerminalSessionState.READY


@pytest.mark.parametrize(
    ("shell", "command"),
    (("powershell", "Set-Location child"), ("cmd", "cd child")),
)
def test_cwd_persists(runtime, tmp_path, shell, command):
    child = tmp_path / "child"
    child.mkdir()
    session = runtime.start_session(shell, cwd=tmp_path)
    changed = runtime.run(session.session_id, command)
    assert Path(changed.cwd) == child.resolve()
    query = runtime.run(session.session_id, "Get-Location" if shell == "powershell" else "cd")
    assert str(child) in query.stdout


@pytest.mark.parametrize(
    ("shell", "set_command", "get_command"),
    (
        ("powershell", "$env:JARVIS_PHASE2='persistent'", "Write-Output $env:JARVIS_PHASE2"),
        ("cmd", "set JARVIS_PHASE2=persistent", "echo %JARVIS_PHASE2%"),
    ),
)
def test_environment_persists(runtime, tmp_path, shell, set_command, get_command):
    session = runtime.start_session(shell, cwd=tmp_path)
    assert runtime.run(session.session_id, set_command).exit_code == 0
    assert "persistent" in runtime.run(session.session_id, get_command).stdout


def test_explicit_environment_and_redacted_snapshot(runtime, tmp_path):
    session = runtime.start_session(
        "powershell",
        cwd=tmp_path,
        environment={"JARVIS_API_TOKEN": "never-display", "JARVIS_COLOR": "blue"},
    )
    assert session.environment == {"JARVIS_API_TOKEN": "<redacted>", "JARVIS_COLOR": "<set>"}
    result = runtime.run(session.session_id, "Write-Output $env:JARVIS_COLOR")
    assert result.stdout.strip() == "blue"


@pytest.mark.parametrize("shell", ("powershell", "cmd"))
def test_stdout_capture_and_success_exit(runtime, tmp_path, shell):
    session = runtime.start_session(shell, cwd=tmp_path)
    command = "Write-Output terminal-ok" if shell == "powershell" else "echo terminal-ok"
    result = runtime.run(session.session_id, command)
    assert result.status is ExecutionState.SUCCEEDED
    assert result.exit_code == 0
    assert "terminal-ok" in result.stdout


def test_stderr_capture_does_not_imply_failure(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    result = runtime.run(session.session_id, "[Console]::Error.WriteLine('warning-only')")
    assert result.status is ExecutionState.SUCCEEDED
    assert result.exit_code == 0
    assert "warning-only" in result.stderr


@pytest.mark.parametrize("shell", ("powershell", "cmd"))
def test_nonzero_exit_is_failure(runtime, tmp_path, shell):
    session = runtime.start_session(shell, cwd=tmp_path)
    result = runtime.run(session.session_id, "cmd /c exit 7")
    assert result.status is ExecutionState.FAILED
    assert result.exit_code == 7
    assert result.error_code is TerminalErrorCode.PROCESS_FAILED


def test_command_not_found_taxonomy(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    result = runtime.run(session.session_id, "jarvis-command-that-does-not-exist")
    assert result.status is ExecutionState.FAILED
    assert result.error_code is TerminalErrorCode.COMMAND_NOT_FOUND


@pytest.mark.parametrize(
    ("shell", "command"),
    (("powershell", "Write-Output 'áéíóú 日本語'"), ("cmd", "echo áéíóú")),
)
def test_utf8_output(runtime, tmp_path, shell, command):
    session = runtime.start_session(shell, cwd=tmp_path)
    result = runtime.run(session.session_id, command)
    assert "áéíóú" in result.stdout


def test_sequential_commands_use_same_pid(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    runtime.run(session.session_id, "Write-Output first")
    runtime.run(session.session_id, "Write-Output second")
    assert runtime.status(session.session_id).pid == session.pid


def test_long_running_returns_without_blocking_and_streams(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    command = "python -u -c \"import time; print('first', flush=True); time.sleep(2); print('last', flush=True)\""
    started = time.perf_counter()
    accepted = runtime.run(session.session_id, command, wait=False, timeout_seconds=10)
    assert time.perf_counter() - started < 0.5
    assert accepted.status is ExecutionState.RUNNING
    time.sleep(0.7)
    partial = runtime.read_output(session.session_id, accepted.execution_id)
    assert "first" in partial.stdout
    assert "last" not in partial.stdout
    completed = runtime.wait(session.session_id, accepted.execution_id, 5)
    assert completed.status is ExecutionState.SUCCEEDED
    assert "last" in completed.stdout


def test_incremental_cursor_has_no_duplicates(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(session.session_id, "Write-Output one; Start-Sleep -Milliseconds 500; Write-Output two", wait=False)
    time.sleep(0.25)
    first = runtime.read_output(session.session_id, accepted.execution_id)
    cursor = max((item.sequence for item in first.output), default=0)
    final = runtime.wait(session.session_id, accepted.execution_id, 3)
    delta = runtime.read_output(session.session_id, accepted.execution_id, after_sequence=cursor)
    assert {item.sequence for item in first.output}.isdisjoint(item.sequence for item in delta.output)
    assert "two" in final.stdout


def test_timeout_is_distinct_and_closes_owned_session(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    result = runtime.run(session.session_id, "Start-Sleep -Seconds 10", timeout_seconds=0.3)
    assert result.status is ExecutionState.TIMED_OUT
    assert result.error_code is TerminalErrorCode.TIMED_OUT
    assert result.duration_ms >= 250
    assert runtime.status(session.session_id).state is TerminalSessionState.FAILED


def test_cancellation_and_cancel_latency(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(session.session_id, "Start-Sleep -Seconds 30", wait=False, timeout_seconds=None)
    started = time.perf_counter()
    result = runtime.cancel(session.session_id, accepted.execution_id)
    assert result.status is ExecutionState.CANCELLED
    assert result.error_code is TerminalErrorCode.CANCELLED
    assert time.perf_counter() - started < 4


def test_cancel_does_not_kill_unrelated_process(runtime, tmp_path):
    external = subprocess.Popen([str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"), "/c", "ping 127.0.0.1 -n 8 >nul"])
    try:
        session = runtime.start_session("powershell", cwd=tmp_path)
        accepted = runtime.run(session.session_id, "Start-Sleep -Seconds 30", wait=False, timeout_seconds=None)
        runtime.cancel(session.session_id, accepted.execution_id)
        assert external.poll() is None
    finally:
        external.terminate()
        external.wait(timeout=3)


def test_cancel_kills_owned_child_process_tree(runtime, tmp_path):
    import psutil

    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(
        session.session_id,
        "python -c \"import time; time.sleep(30)\"",
        wait=False,
        timeout_seconds=None,
    )
    time.sleep(0.4)
    descendants = psutil.Process(session.pid).children(recursive=True)
    assert descendants
    runtime.cancel(session.session_id, accepted.execution_id)
    _, alive = psutil.wait_procs(descendants, timeout=3)
    assert not alive


def test_session_close_and_command_rejected(runtime, tmp_path):
    session = runtime.start_session("cmd", cwd=tmp_path)
    closed = runtime.close_session(session.session_id)
    assert closed.state is TerminalSessionState.CLOSED
    with pytest.raises(TerminalError) as captured:
        runtime.run(session.session_id, "echo no")
    assert captured.value.code is TerminalErrorCode.SESSION_CLOSED


def test_invalid_cwd(runtime, tmp_path):
    with pytest.raises(TerminalError) as captured:
        runtime.start_session("powershell", cwd=tmp_path / "missing")
    assert captured.value.code is TerminalErrorCode.INVALID_CWD


def test_output_truncation_is_explicit(tmp_path):
    runtime = TerminalRuntime(max_output_chars=1024)
    try:
        session = runtime.start_session("powershell", cwd=tmp_path)
        result = runtime.run(session.session_id, "Write-Output ('x' * 3000)")
        assert result.truncated
        assert len(result.stdout) <= 1024
    finally:
        runtime.shutdown()


def test_duplicate_session_ids_are_impossible(runtime, tmp_path):
    ids = {runtime.start_session("cmd", cwd=tmp_path).session_id for _ in range(8)}
    assert len(ids) == 8


def test_shutdown_cleans_owned_shell(runtime, tmp_path):
    import psutil

    session = runtime.start_session("powershell", cwd=tmp_path)
    pid = session.pid
    runtime.shutdown()
    assert not psutil.pid_exists(pid)


def test_concurrent_executions_in_separate_sessions(runtime, tmp_path):
    first = runtime.start_session("powershell", cwd=tmp_path)
    second = runtime.start_session("cmd", cwd=tmp_path)
    a = runtime.run(first.session_id, "Start-Sleep -Milliseconds 400; Write-Output a", wait=False)
    b = runtime.run(second.session_id, "ping 127.0.0.1 -n 2 >nul & echo b", wait=False)
    assert runtime.wait(first.session_id, a.execution_id, 3).status is ExecutionState.SUCCEEDED
    assert runtime.wait(second.session_id, b.execution_id, 3).status is ExecutionState.SUCCEEDED


def test_same_session_rejects_concurrent_command(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(session.session_id, "Start-Sleep -Seconds 5", wait=False, timeout_seconds=None)
    with pytest.raises(TerminalError) as captured:
        runtime.run(session.session_id, "Write-Output no")
    assert captured.value.code is TerminalErrorCode.SESSION_BUSY
    runtime.cancel(session.session_id, accepted.execution_id)


def test_controlled_stdin(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(
        session.session_id,
        "python -u -c \"print('got:'+input(), flush=True)\"",
        wait=False,
        accepts_input=True,
        timeout_seconds=5,
    )
    runtime.write(session.session_id, accepted.execution_id, "answer")
    result = runtime.wait(session.session_id, accepted.execution_id, 5)
    assert result.status is ExecutionState.SUCCEEDED
    assert "got:answer" in result.stdout


def test_stdin_rejected_unless_declared(runtime, tmp_path):
    session = runtime.start_session("powershell", cwd=tmp_path)
    accepted = runtime.run(session.session_id, "Start-Sleep -Seconds 5", wait=False, timeout_seconds=None)
    with pytest.raises(TerminalError) as captured:
        runtime.write(session.session_id, accepted.execution_id, "unsafe")
    assert captured.value.code is TerminalErrorCode.INPUT_NOT_ACCEPTED
    runtime.cancel(session.session_id, accepted.execution_id)


def test_venv_python_is_exact(runtime):
    root = Path(__file__).resolve().parents[1]
    venv = root / ".venv"
    session = runtime.start_session("powershell", cwd=root, venv_path=venv)
    result = runtime.run(session.session_id, "python -c \"import sys; print(sys.executable)\"")
    assert str(venv / "Scripts" / "python.exe").lower() in result.stdout.lower()


def test_activity_events_are_real_and_do_not_contain_command(tmp_path):
    events = []
    runtime = TerminalRuntime(event_sink=lambda name, payload: events.append((name, payload)))
    try:
        session = runtime.start_session("powershell", cwd=tmp_path)
        runtime.run(session.session_id, "Write-Output private-command-text")
        runtime.close_session(session.session_id)
    finally:
        runtime.shutdown()
    names = {name for name, _ in events}
    assert {"terminal.session_started", "terminal.command_started", "terminal.command_completed", "terminal.session_closed"} <= names
    assert "private-command-text" not in repr(events)


def test_terminal_actions_and_tools_keep_dotted_names(runtime):
    module = TerminalModule(runtime)
    actions = ActionRegistry()
    module.register_actions(actions)
    assert {item.name for item in actions.list_actions()} == {
        "terminal.start_session", "terminal.run", "terminal.write", "terminal.read_output",
        "terminal.status", "terminal.cancel", "terminal.close_session", "terminal.open_visible",
    }

    class Collector:
        def __init__(self):
            self.names = []

        def register(self, definition):
            self.names.append(definition.name)

    collector = Collector()
    module.register_tools(collector)
    assert set(collector.names) == {item.name for item in actions.list_actions()}


def test_security_gate_allows_safe_terminal_command_and_blocks_dangerous(runtime):
    module = TerminalModule(runtime)
    actions = ActionRegistry()
    module.register_actions(actions)
    definition = actions.get("terminal.run")
    assert definition is not None and definition.risk is ActionRisk.SENSITIVE
    gate = TypedActionSecurityGate()
    safe = gate.evaluate(definition, ActionRequest(run_id="r", call_id="safe", action="terminal.run", arguments={"session_id": "s", "command": "python --version"}))
    gpu = gate.evaluate(definition, ActionRequest(run_id="r", call_id="gpu", action="terminal.run", arguments={"session_id": "s", "command": "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"}))
    blocked = gate.evaluate(definition, ActionRequest(run_id="r", call_id="bad", action="terminal.run", arguments={"session_id": "s", "command": "shutdown /s"}))
    sensitive = gate.evaluate(definition, ActionRequest(run_id="r", call_id="ask", action="terminal.run", arguments={"session_id": "s", "command": "pip install package"}))
    assert safe.status is SecurityDecisionStatus.ALLOWED
    assert gpu.status is SecurityDecisionStatus.ALLOWED
    assert blocked.status is SecurityDecisionStatus.BLOCKED
    assert sensitive.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED


def test_security_gate_confirms_interpreter_code_execution(runtime):
    """python -c / -m and interpreter paths must ask for confirmation (audit P1.1)."""
    module = TerminalModule(runtime)
    actions = ActionRegistry()
    module.register_actions(actions)
    definition = actions.get("terminal.run")
    assert definition is not None
    gate = TypedActionSecurityGate()

    code_exec = gate.evaluate(definition, ActionRequest(run_id="r", call_id="c1", action="terminal.run", arguments={"session_id": "s", "command": "python -c \"__import__('os').system('calc')\""}))
    module_server = gate.evaluate(definition, ActionRequest(run_id="r", call_id="c2", action="terminal.run", arguments={"session_id": "s", "command": "python -m http.server"}))
    absolute_interpreter = gate.evaluate(definition, ActionRequest(run_id="r", call_id="c3", action="terminal.run", arguments={"session_id": "s", "command": "C:\\Tools\\python.exe -c \"import os\""}))
    version_check = gate.evaluate(definition, ActionRequest(run_id="r", call_id="c4", action="terminal.run", arguments={"session_id": "s", "command": "python --version"}))
    allowlisted_module = gate.evaluate(definition, ActionRequest(run_id="r", call_id="c5", action="terminal.run", arguments={"session_id": "s", "command": "python -m pytest -q"}))

    assert code_exec.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED
    assert module_server.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED
    assert absolute_interpreter.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED
    assert version_check.status is SecurityDecisionStatus.ALLOWED
    assert allowlisted_module.status is SecurityDecisionStatus.ALLOWED


def test_tool_payload_contracts_are_compact():
    assert StartSessionPayload(shell="cmd").model_dump()["environment"] == {}
    schema = RunTerminalPayload.model_json_schema()
    assert {"session_id", "command"} <= set(schema["properties"])
