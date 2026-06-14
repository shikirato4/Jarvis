from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.memory.project_memory import ProjectMemory
from jarvis.cli import app


def test_project_memory_initializes_when_missing(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "runtime" / "code_agent" / "project_memory.json", project_root=tmp_path)

    data = memory.load()

    assert data["project_name"] == tmp_path.name
    assert data["project_root"] == str(tmp_path.resolve(strict=False))
    assert data["security_rules"]


def test_project_memory_loads_existing_file(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)
    memory.initialize(project_name="Jarvis")

    loaded = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path).load()

    assert loaded["project_name"] == "Jarvis"


def test_project_memory_recovers_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("{broken json", encoding="utf-8")
    memory = ProjectMemory(path, project_root=tmp_path)

    data = memory.load()

    assert data["project_name"] == tmp_path.name
    assert data["warnings"]
    assert "corrupt" in data["last_activity"]["action"]
    assert list(tmp_path.glob("memory.json.corrupt-*.bak"))


def test_project_memory_records_decisions_issues_and_phase(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)

    memory.add_decision("Use JSON memory", "Simple local storage", ["src/app.py"])
    memory.add_fixed_issue("Blocked direct runner", "Tool bypass", "AuthorizationContext", ["terminal_runner.py"])
    data = memory.add_phase_completed("Fase 3 - Memoria de proyecto")

    assert data["technical_decisions"][0]["title"] == "Use JSON memory"
    assert data["fixed_issues"][0]["fix"] == "AuthorizationContext"
    assert data["completed_phases"][0]["name"] == "Fase 3 - Memoria de proyecto"


def test_project_memory_records_and_completes_pending_tasks(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)

    data = memory.add_pending_task("Add CLI memory commands", priority="high")
    task_id = data["pending_tasks"][0]["id"]
    completed = memory.complete_task(task_id)

    assert completed["pending_tasks"][0]["status"] == "completed"
    assert completed["pending_tasks"][0]["completed_at"]


def test_project_memory_context_summary_is_short_and_useful(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)
    memory.add_decision("Keep CodeAgentExecutor as gate", "Tools need authorization context", [])
    memory.add_pending_task("Run related tests")

    summary = memory.get_agent_context_summary(max_chars=500)

    assert "Project:" in summary
    assert "Keep CodeAgentExecutor as gate" in summary
    assert "Run related tests" in summary
    assert len(summary) <= 500


def test_project_memory_redacts_obvious_secrets(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)

    data = memory.add_note("Do not store token abc123, password abc, api key xyz, private key, certificate.pem or .env contents")

    assert "[redacted]" in data["user_notes"][0]["text"]


def test_project_memory_does_not_store_dangerous_command_as_executable_tip(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)

    data = memory.add_useful_command("rm -rf .", "never recommend", risk_level=3)

    assert data["useful_commands"][0]["command"] == "[redacted]"


def test_project_memory_summary_redacts_sensitive_content_and_respects_limit(tmp_path: Path) -> None:
    memory = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path)
    memory.add_decision("Do not store .env token", "password and credentials must not appear", [])

    summary = memory.get_agent_context_summary(max_chars=120)

    assert len(summary) <= 120
    assert ".env" not in summary
    assert "token" not in summary.casefold()
    assert "password" not in summary.casefold()


def test_project_memory_scan_ignores_heavy_and_sensitive_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    for dirname in (".git", "node_modules", "dist", "build", ".next", "__pycache__", "venv", ".venv"):
        folder = tmp_path / dirname
        folder.mkdir()
        (folder / "large.py").write_text("noise", encoding="utf-8")

    summary = ProjectMemory(tmp_path / "memory.json", project_root=tmp_path).scan_project()

    assert "src/app.py" in {path.replace("\\", "/") for path in summary["key_files"]}
    assert all(".env" not in path for path in summary["key_files"])
    ignored_names = (".git", "node_modules", "dist", "build", ".next", "__pycache__", "venv", ".venv")
    assert all(not any(name in path for name in ignored_names) for path in summary["top_directories"])


def test_code_agent_executor_updates_memory_after_actions(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    service.scan_project()
    service.write_file("notes.txt", "hello")
    service.run_command("git status", dry_run=True)
    memory = service.memory_show()

    assert memory["project_structure"]
    assert any("Modified file" in note["text"] for note in memory["user_notes"])
    assert any(command["command"] == "git status" for command in memory["useful_commands"])


def test_code_memory_cli_commands(tmp_path: Path) -> None:
    runner = CliRunner()

    show = runner.invoke(app, ["code", "memory", "show", "--root", str(tmp_path)])
    summary = runner.invoke(app, ["code", "memory", "summary", "--root", str(tmp_path), "--max-chars", "300"])
    note = runner.invoke(app, ["code", "memory", "add-note", "remember this", "--root", str(tmp_path)])
    task = runner.invoke(app, ["code", "memory", "add-task", "finish tests", "--root", str(tmp_path)])
    scan = runner.invoke(app, ["code", "memory", "scan-project", "--root", str(tmp_path)])
    phase = runner.invoke(app, ["code", "memory", "add-phase", "Fase 3", "--root", str(tmp_path)])

    assert show.exit_code == 0
    assert summary.exit_code == 0
    assert "Project:" in summary.stdout
    assert note.exit_code == 0
    assert task.exit_code == 0
    assert scan.exit_code == 0
    assert phase.exit_code == 0

    task_id = CodeAgentRuntimeService(tmp_path).memory_show()["pending_tasks"][0]["id"]
    completed = runner.invoke(app, ["code", "memory", "complete-task", task_id, "--root", str(tmp_path)])

    assert completed.exit_code == 0
    assert '"status": "completed"' in completed.stdout
