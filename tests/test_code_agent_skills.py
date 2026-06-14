from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.skills import CodeAgentSkill, SkillRegistry, build_builtin_registry
from jarvis.code_agent_runtime.skills.builtin_skills import builtin_skills
from jarvis.code_agent_runtime.tools.terminal_runner import TerminalRunner


def _ids(payload: dict) -> list[str]:
    return [item["id"] for item in payload["suggested_skills"]]


def test_builtin_skills_register_and_list() -> None:
    registry = build_builtin_registry()

    ids = registry.ids()

    assert "python" in ids
    assert "testing" in ids
    assert "security-audit" in ids
    assert "frontend-react" in ids


def test_registry_rejects_duplicate_ids() -> None:
    registry = SkillRegistry()
    skill = CodeAgentSkill(id="python", name="Python", description="Python workflow")

    registry.register(skill)
    with pytest.raises(ValueError):
        registry.register(skill)


def test_registry_get_and_search_by_tag() -> None:
    registry = build_builtin_registry()

    assert registry.get("python").id == "python"
    assert "security-audit" in {skill.id for skill in registry.by_tag("security")}


def test_skill_router_suggests_python_testing_debugging(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_suggest("Arregla este error de pytest con traceback")
    ids = _ids(payload)

    assert "testing" in ids
    assert "debugging" in ids
    assert "python" in ids


def test_skill_router_suggests_security_audit_for_permissions(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_suggest("Audita permisos, path traversal ../ y command injection")
    ids = _ids(payload)

    assert "security-audit" in ids
    assert "testing" in ids


def test_skill_router_suggests_git_review_for_diff_status_checkpoint(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_suggest("Revisa git status, diff y checkpoint antes de seguir")
    ids = _ids(payload)

    assert "git-review" in ids
    assert "testing" in ids


def test_skill_router_suggests_frontend_react(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_suggest("Haz un componente visual en React TypeScript")
    ids = _ids(payload)

    assert "frontend-react" in ids


def test_skill_context_generates_strategy_without_executing_commands(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    context = service.skills_context("Arregla errores de pytest", limit=3)
    memory = service.memory_show()

    assert context["skill_contexts"]
    assert any(item["skill_id"] == "testing" for item in context["skill_contexts"])
    assert context["skill_contexts"][0]["safe_commands_suggested"]
    assert memory["useful_commands"] == []
    assert memory["failed_commands"] == []


def test_skills_do_not_execute_commands(monkeypatch, tmp_path: Path) -> None:
    def fail_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("skills must not execute commands")

    monkeypatch.setattr(TerminalRunner, "run", fail_run)
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_context("Arregla errores de pytest", limit=3)

    assert payload["skill_contexts"]


def test_skill_suggestions_are_recorded_in_memory_without_secrets(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    service.skills_suggest("Audita token secret password .env command injection")
    memory = service.memory_show()
    raw = str(memory).casefold()

    assert memory["skill_suggestions"]
    assert "[redacted]" in raw
    assert "token secret password" not in raw


def test_skill_context_output_redacts_secret_task_content(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_context("Audita .env token password credentials API key private key certificate", limit=3)
    raw = str(payload).casefold()

    assert "[redacted]" in raw
    for secret in (".env", "token", "password", "credentials", "api key", "private key", "certificate"):
        assert secret not in raw


def test_frontend_react_does_not_activate_from_project_context_only(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)
    service.executor.project_memory.update_project_structure({"key_files": ["src/App.tsx", "package.json"], "languages": {".tsx": 1}})

    payload = service.skills_suggest("Audita permisos y command injection")
    ids = _ids(payload)

    assert "security-audit" in ids
    assert "frontend-react" not in ids


def test_skill_context_has_reasonable_output_limit(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    payload = service.skills_context("Audita permisos y command injection", limit=5, max_memory_chars=300)

    assert len(payload["memory_summary"]) <= 300
    assert len(str(payload)) < 30_000


def test_builtin_skill_suggested_commands_are_non_destructive() -> None:
    dangerous = ("rm -rf", "del /s", "rmdir /s", "sudo", "curl", "| sh", "wget", "| bash", "git push", "git reset --hard", "git clean -fd")

    for skill in builtin_skills():
        for command in skill.safe_commands:
            folded = command.casefold()
            assert not any(pattern in folded for pattern in dangerous), (skill.id, command)


def test_skills_cli_list_show_suggest_context(tmp_path: Path) -> None:
    runner = CliRunner()

    listed = runner.invoke(app, ["code", "skills", "list", "--root", str(tmp_path)])
    shown = runner.invoke(app, ["code", "skills", "show", "python", "--root", str(tmp_path)])
    suggested = runner.invoke(app, ["code", "skills", "suggest", "arregla errores de pytest", "--root", str(tmp_path)])
    context = runner.invoke(app, ["code", "skills", "context", "audita permisos", "--root", str(tmp_path)])

    assert listed.exit_code == 0
    assert '"id": "python"' in listed.stdout
    assert shown.exit_code == 0
    assert '"safe_commands"' in shown.stdout
    assert suggested.exit_code == 0
    assert "testing" in suggested.stdout
    assert context.exit_code == 0
    assert "skill_contexts" in context.stdout


def test_skills_cli_by_tag(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["code", "skills", "by-tag", "security", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "security-audit" in result.stdout
