from __future__ import annotations

import inspect
import os
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.change_generator import ChangeGenerator


def _make_project(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "tests").mkdir()
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (root / "src" / "other.py").write_text("VALUE = 'hello'\n", encoding="utf-8")
    (root / "tests" / "test_app.py").write_text("from src.app import greet\n", encoding="utf-8")
    (root / "package.json").write_text('{"dependencies": {}}\n', encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")


def test_change_targets_explicit_file_detected(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_targets("cambia hello por hola en src/app.py")

    assert result["status"] == "resolved"
    assert result["targets"][0]["path"].replace("\\", "/") == "src/app.py"


def test_change_targets_handles_quotes_punctuation_and_absolute_inside_root(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    quoted = service.change_targets('cambia "hello" por "hola" en "src/app.py".')
    absolute = service.change_targets(f"cambia hello por hola en {tmp_path / 'src' / 'app.py'}")

    assert quoted["status"] == "resolved"
    assert quoted["targets"][0]["path"].replace("\\", "/") == "src/app.py"
    assert absolute["status"] == "resolved"
    assert absolute["targets"][0]["path"].replace("\\", "/") == "src/app.py"


def test_change_targets_and_plan_do_not_create_memory_or_modify_files(tmp_path: Path) -> None:
    _make_project(tmp_path)
    readme = tmp_path / "README.md"
    before = readme.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)
    memory_path = tmp_path / "runtime" / "code_agent" / "project_memory.json"

    targets = service.change_targets("agrega una nota al README")
    plan = service.change_plan("agrega una nota al README")

    assert targets["status"] == "resolved"
    assert plan["status"] == "proposed"
    assert readme.read_text(encoding="utf-8") == before
    assert not memory_path.exists()


def test_change_ambiguous_task_needs_review(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("arregla error de pytest")

    assert result["status"] == "needs_review"
    assert not result["patch_id"]


def test_change_blocks_sensitive_and_outside_targets(tmp_path: Path) -> None:
    _make_project(tmp_path)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    sensitive = service.change_propose("cambia TOKEN por X en .env")
    outside_result = service.change_propose(f"cambia x por y en {outside}")

    assert sensitive["status"] == "blocked"
    assert outside_result["status"] == "blocked"


def test_change_blocks_traversal_symlink_and_protected_project_files(tmp_path: Path) -> None:
    _make_project(tmp_path)
    outside = tmp_path.parent / "outside-link-target.py"
    outside.write_text("hello\n", encoding="utf-8")
    link = tmp_path / "src" / "linked.py"
    try:
        os.symlink(outside, link)
    except OSError:
        link = None
    service = CodeAgentRuntimeService(tmp_path)

    traversal = service.change_propose("cambia hello por hola en ../outside.py")
    protected = service.change_propose("agrega una dependencia al package.json")
    symlink = service.change_propose("cambia hello por hola en src/linked.py") if link else {"status": "blocked"}

    assert traversal["status"] == "blocked"
    assert protected["status"] == "blocked"
    assert symlink["status"] == "blocked"


def test_change_readme_append_generates_patch_without_applying(tmp_path: Path) -> None:
    _make_project(tmp_path)
    readme = tmp_path / "README.md"
    before = readme.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("agrega una nota al README")

    assert result["status"] == "proposed"
    assert result["patch_id"]
    assert "unified_diff" in result["patch"]
    assert readme.read_text(encoding="utf-8") == before


def test_change_exact_replace_generates_patch(tmp_path: Path) -> None:
    _make_project(tmp_path)
    target = tmp_path / "src" / "app.py"
    before = target.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("cambia hello por hola en src/app.py")

    assert result["status"] == "proposed"
    assert "+    return 'hola'" in result["patch"]["unified_diff"]
    assert target.read_text(encoding="utf-8") == before


def test_change_missing_old_text_returns_blocked_or_review(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("cambia missing por hola en src/app.py")

    assert result["status"] in {"blocked", "needs_review"}
    assert "old_text" in str(result).casefold() or not result["patch_id"]


def test_change_anchor_or_unsupported_operations_need_review(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    anchor = service.change_propose("inserta texto despues de missing-anchor en src/app.py")
    delete = service.change_propose("borra archivo src/app.py")

    assert anchor["status"] == "needs_review"
    assert delete["status"] == "needs_review"


def test_change_create_file_generates_patch_without_creating_file(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose('crea archivo docs/usage.md con contenido "Uso seguro de Jarvis."')

    assert result["status"] == "proposed"
    assert [item.replace("\\", "/") for item in result["patch"]["target_files"]] == ["docs/usage.md"]
    assert not (tmp_path / "docs" / "usage.md").exists()


def test_change_large_refactor_needs_review(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("refactoriza toda la app")

    assert result["status"] == "needs_review"
    assert not result["patch_id"]


def test_agent_generate_patch_does_not_apply_without_explicit_patch_id(tmp_path: Path) -> None:
    _make_project(tmp_path)
    target = tmp_path / "README.md"
    before = target.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    dry_run = service.agent_run("agrega una nota al README", mode="dry-run", generate_patch=True)
    assisted = service.agent_run("agrega una nota al README", mode="assisted", generate_patch=True)
    apply = service.agent_run("agrega una nota al README", mode="apply", generate_patch=True, confirm=True)

    assert dry_run["patch"]["status"] == "proposed"
    assert assisted["patch"]["status"] == "proposed"
    assert apply["patch"]["status"] == "proposed"
    assert target.read_text(encoding="utf-8") == before
    assert dry_run["commands"] == []
    assert assisted["commands"] == []
    assert apply["commands"] == []
    assert "not applied automatically" in str(apply).casefold()


def test_change_memory_sanitizes_secrets(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    service.change_propose("agrega token password .env al README")
    raw = str(service.memory_show()).casefold()

    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "token password" not in raw


def test_change_generator_does_not_use_unsafe_tools_directly() -> None:
    source = inspect.getsource(ChangeGenerator)

    assert "FileWriter(" not in source
    assert "TerminalRunner(" not in source
    assert ".writer.write_text(" not in source
    assert ".runner.run(" not in source


def test_change_cli_targets_plan_propose_and_agent_generate_patch(tmp_path: Path) -> None:
    _make_project(tmp_path)
    runner = CliRunner()

    targets = runner.invoke(app, ["code", "change", "targets", "cambia hello por hola en src/app.py", "--root", str(tmp_path)])
    plan = runner.invoke(app, ["code", "change", "plan", "cambia hello por hola en src/app.py", "--root", str(tmp_path)])
    propose = runner.invoke(app, ["code", "change", "propose", "agrega una nota al README", "--root", str(tmp_path)])
    agent = runner.invoke(app, ["code", "agent", "run", "agrega una nota al README", "--root", str(tmp_path), "--dry-run", "--generate-patch"])
    assisted = runner.invoke(app, ["code", "agent", "run", "agrega una nota al README", "--root", str(tmp_path), "--assisted", "--generate-patch"])
    apply = runner.invoke(app, ["code", "agent", "run", "agrega una nota al README", "--root", str(tmp_path), "--apply", "--generate-patch"])

    assert targets.exit_code == 0
    assert '"src' in targets.stdout
    assert plan.exit_code == 0
    assert '"operations"' in plan.stdout
    assert propose.exit_code == 0
    assert '"proposed"' in propose.stdout
    assert agent.exit_code == 0
    assert '"patch_id"' in agent.stdout
    assert assisted.exit_code == 0
    assert '"commands": []' in assisted.stdout
    assert apply.exit_code == 0
    assert '"commands": []' in apply.stdout
