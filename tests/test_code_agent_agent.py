from __future__ import annotations

import inspect
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime.agent import AgentRunner, AgentVerifier, ExecutionPlan, PlanStep, PlanStepStatus
from jarvis.code_agent_runtime.agent.models import AgentRunMode, AgentTask
from jarvis.code_agent_runtime.base import RiskLevel
from jarvis.code_agent_runtime import CodeAgentRuntimeService


def _make_project(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text("from src.app import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")


def _make_repo_library(root: Path) -> Path:
    library = root / "repos"
    repo = library / "patterns"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "README.md").write_text("Python pytest CLI security patterns", encoding="utf-8")
    (repo / "src" / "security.py").write_text("def validate_path(path):\n    return 'path traversal command injection permissions'\n", encoding="utf-8")
    (repo / "tests" / "test_security.py").write_text("import pytest\n\ndef test_security():\n    assert True\n", encoding="utf-8")
    return library


def test_agent_plan_for_python_testing_task(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.agent_plan("arregla errores de pytest")
    plan = result["plan"]

    assert plan["task"]["task_type"] == "testing"
    assert {"python", "testing"}.intersection(set(plan["task"]["skills"]))
    assert any(step["action_type"] == "run_command" for step in plan["steps"])
    assert any(step["tool"] == "local_search" for step in plan["steps"])


def test_agent_plan_for_security_and_cli_tasks(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    security = service.agent_plan("audita permisos path traversal command injection")
    cli = service.agent_plan("agrega comando CLI con Typer")

    assert security["plan"]["task"]["task_type"] == "security-audit"
    assert "security-audit" in security["plan"]["task"]["skills"]
    assert cli["plan"]["task"]["task_type"] == "cli"
    assert "cli" in cli["plan"]["task"]["skills"]


def test_agent_context_builder_uses_search_memory_git_and_sanitizes_secrets(tmp_path: Path) -> None:
    _make_project(tmp_path)
    library = _make_repo_library(tmp_path.parent / "library")
    service = CodeAgentRuntimeService(tmp_path)
    service.repos_index(str(library))
    service.learn_extract()
    service.local_search_rebuild()

    context = service.agent_context("arregla pytest con token password .env")
    raw = str(context).casefold()

    assert context["memory_summary"]
    assert context["search_context"]
    assert context["git_summary"]
    assert context["project_structure"]
    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "token password" not in raw
    assert len(context["context_summary"]) <= 6000


def test_agent_dry_run_does_not_modify_files_or_execute_commands(tmp_path: Path) -> None:
    _make_project(tmp_path)
    target = tmp_path / "src" / "app.py"
    before = target.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    result = service.agent_run("arregla errores de pytest", mode="dry-run")

    assert target.read_text(encoding="utf-8") == before
    assert result["mode"] == "dry-run"
    assert all(step["status"] == "skipped" for step in result["plan"]["steps"])
    assert result["commands"] == []


def test_agent_assisted_blocks_sensitive_actions_without_confirmation(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.agent_run("instala paquete npm install", mode="assisted")

    assert result["verification"]["status"] == "blocked"
    assert any(step["status"] == "blocked" for step in result["plan"]["steps"])
    assert "requires explicit confirmation" in str(result).casefold()


def test_agent_apply_uses_existing_permissions_for_safe_commands(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.agent_run("arregla errores de pytest", mode="apply", max_commands=1)

    assert result["mode"] == "apply"
    assert any("python -m pytest" in command for command in result["commands"])
    assert result["verification"]["status"] in {"success", "blocked", "failed"}


def test_agent_blocks_missing_pin_for_sensitive_install(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.agent_run("instala paquete npm install", mode="apply", confirm=True)

    assert result["verification"]["status"] == "blocked"
    assert any(step["status"] == "blocked" and "PIN" in step["result_summary"] for step in result["plan"]["steps"])
    assert result["commands"] == []


def test_agent_dangerous_commands_still_go_through_permission_gate(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    step = PlanStep(
        id="danger",
        description="dangerous command probe",
        action_type="run_command",
        tool="terminal_runner",
        risk_level=RiskLevel.CRITICAL,
        requires_confirmation=True,
        requires_pin=True,
        data={"command": "rm -rf ."},
    )
    plan = ExecutionPlan(
        task=AgentTask(original_text="danger", objective="danger", task_type="security-audit", max_steps=1),
        mode=AgentRunMode.APPLY,
        steps=[step],
    )
    service.executor.agent._execute_step(step, mode=AgentRunMode.APPLY, confirm=True, pin="1234", touched_files=[], commands=[], errors=[])

    assert plan.steps[0].status == PlanStepStatus.BLOCKED
    assert plan.steps[0].data["status"] == "blocked"


def test_agent_limits_steps_and_commands(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    planned = service.agent_plan("arregla errores de pytest y compile", max_steps=4)
    run = service.agent_run("arregla errores de pytest y compile", mode="apply", max_steps=4, max_commands=0)

    assert len(planned["plan"]["steps"]) <= 4
    assert len(run["plan"]["steps"]) <= 4
    assert run["commands"] == []


def test_agent_runner_source_does_not_use_unsafe_tools_directly() -> None:
    source = inspect.getsource(AgentRunner)

    assert "TerminalRunner(" not in source
    assert "FileWriter(" not in source
    assert ".runner.run(" not in source
    assert ".writer.write_text(" not in source


def test_agent_verifier_reports_all_statuses() -> None:
    verifier = AgentVerifier()
    task = AgentTask(original_text="x", objective="x", task_type="programming")

    success = verifier.verify(ExecutionPlan(task=task, steps=[PlanStep(id="1", description="ok", action_type="x", tool="x", status=PlanStepStatus.DONE)]))
    blocked = verifier.verify(ExecutionPlan(task=task, steps=[PlanStep(id="1", description="blocked", action_type="x", tool="x", status=PlanStepStatus.BLOCKED)]))
    failed = verifier.verify(ExecutionPlan(task=task, steps=[PlanStep(id="1", description="failed", action_type="x", tool="x", status=PlanStepStatus.FAILED)]))
    partial = verifier.verify(ExecutionPlan(task=task, steps=[PlanStep(id="1", description="pending", action_type="x", tool="x")]))

    assert success.status == "success"
    assert blocked.status == "blocked"
    assert failed.status == "failed"
    assert partial.status == "partial"


def test_agent_records_memory_without_secrets(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    service.agent_run("arregla token password .env pytest", mode="dry-run")
    memory = service.memory_show()
    raw = str(memory).casefold()

    assert memory["agent_events"]
    assert memory["agent_events"][-1]["plan_steps"]
    assert "search_count" in memory["agent_events"][-1]
    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "token password" not in raw


def test_agent_cli_plan_context_run_verify(tmp_path: Path) -> None:
    _make_project(tmp_path)
    runner = CliRunner()

    plan = runner.invoke(app, ["code", "agent", "plan", "arregla errores de pytest", "--root", str(tmp_path)])
    context = runner.invoke(app, ["code", "agent", "context", "arregla errores de pytest", "--root", str(tmp_path)])
    dry_run = runner.invoke(app, ["code", "agent", "run", "arregla errores de pytest", "--root", str(tmp_path), "--dry-run"])
    assisted = runner.invoke(app, ["code", "agent", "run", "arregla errores de pytest", "--root", str(tmp_path), "--assisted"])
    verify = runner.invoke(app, ["code", "agent", "verify", "--root", str(tmp_path)])

    assert plan.exit_code == 0
    assert '"steps"' in plan.stdout
    assert context.exit_code == 0
    assert "context_summary" in context.stdout
    assert dry_run.exit_code == 0
    assert '"dry-run"' in dry_run.stdout
    assert assisted.exit_code == 0
    assert verify.exit_code == 0
    assert '"status"' in verify.stdout
