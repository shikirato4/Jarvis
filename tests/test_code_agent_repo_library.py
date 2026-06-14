from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService


def _make_python_repo(root: Path) -> Path:
    repo = root / "python-agent"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("Python agent with pytest fixtures", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='python-agent'\n", encoding="utf-8")
    (repo / "src" / "agent.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (repo / "tests" / "test_agent.py").write_text("import pytest\n\ndef test_run():\n    assert True\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "noise.py").write_text("pytest secret", encoding="utf-8")
    return repo


def _make_react_repo(root: Path) -> Path:
    repo = root / "react-ui"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("React TypeScript UI components", encoding="utf-8")
    (repo / "package.json").write_text('{"dependencies":{"react":"latest"},"scripts":{"build":"vite build"}}', encoding="utf-8")
    (repo / "src" / "Button.tsx").write_text("export function Button(){ return <button>Save</button> }\n", encoding="utf-8")
    return repo


def _make_security_repo(root: Path) -> Path:
    repo = root / "security-tools"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("Security permissions audit", encoding="utf-8")
    (repo / "src" / "permissions.py").write_text("def check_path(path):\n    # path traversal and command injection audit\n    return 'permission security auth'\n", encoding="utf-8")
    return repo


def test_repo_library_indexes_repos_and_ignores_sensitive_and_heavy_dirs(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_python_repo(library)
    _make_react_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    result = service.repos_index(str(library))
    listing = service.repos_list()
    raw = str(listing).casefold()

    assert result["repo_count"] == 2
    assert result["snippet_count"] > 0
    assert {repo["id"] for repo in listing["repos"]} == {"python-agent", "react-ui"}
    assert ".env" not in raw
    assert "token=secret" not in raw
    assert "node_modules" not in raw


def test_repo_library_keeps_external_repos_read_only(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    repo = _make_python_repo(library)
    tracked_file = repo / "src" / "agent.py"
    before = tracked_file.read_text(encoding="utf-8")
    before_paths = {path.relative_to(repo) for path in repo.rglob("*")}
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    service.repos_search_task("arregla pytest")

    after_paths = {path.relative_to(repo) for path in repo.rglob("*")}
    assert tracked_file.read_text(encoding="utf-8") == before
    assert after_paths == before_paths


def test_repo_library_search_by_keyword_and_skill_task(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_python_repo(library)
    _make_react_repo(library)
    _make_security_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")
    service.repos_index(str(library))

    pytest_results = service.repos_search("pytest fixtures", skill_ids=["python", "testing"])
    task_results = service.repos_search_task("arregla errores de pytest")
    security_results = service.repos_search_task("audita permisos path traversal command injection")
    react_results = service.repos_search_task("haz componente React TypeScript")
    cli_results = service.repos_search_task("agrega comando CLI con Typer")
    git_results = service.repos_search_task("revisa git diff status checkpoint")

    assert pytest_results["results"]
    assert pytest_results["results"][0]["score"] > 0
    assert "matched keywords" in pytest_results["results"][0]["reason"]
    assert any("python-agent" == item["repo_id"] for item in task_results["results"])
    assert any("security-tools" == item["repo_id"] for item in security_results["results"])
    assert any("react-ui" == item["repo_id"] for item in react_results["results"])
    assert {"frontend-react", "testing"}.issubset(set(react_results["skill_ids"]))
    assert {"cli", "testing"}.issubset(set(cli_results["skill_ids"]))
    assert {"git-review", "testing"}.issubset(set(git_results["skill_ids"]))
    assert all("reference only" in item["notice"] for item in pytest_results["results"])


def test_repo_library_sanitizes_secret_snippets(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    repo = library / "secret-example"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("README mentions token password private key", encoding="utf-8")
    (repo / "src" / "main.py").write_text("API_KEY = 'secret-token'\n", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    result = service.repos_search("token password api_key")
    raw = str(result).casefold()

    assert "[redacted]" in raw
    assert "secret-token" not in raw
    assert "private key" not in raw


def test_repo_library_limits_snippets_files_and_results(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    repo = library / "big-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "README.md").write_text("pytest " * 5000, encoding="utf-8")
    (repo / "src" / "large.py").write_text("pytest\n" * 200_000, encoding="utf-8")
    for index in range(80):
        (repo / "src" / f"file_{index}.py").write_text(f"def test_{index}():\n    assert True\n", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    listed = service.repos_list()
    results = service.repos_search("pytest assert", limit=3)

    assert listed["repo_count"] == 1
    assert results["result_count"] <= 3
    assert all(len(item["snippet"]) <= 1200 for item in results["results"])
    raw = str(service.repos_show("big-repo"))
    assert "large.py" not in raw


def test_repo_library_does_not_execute_external_repo_commands(monkeypatch, tmp_path: Path) -> None:
    def fail_subprocess(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("repo library must not execute external repo commands")

    monkeypatch.setattr(subprocess, "run", fail_subprocess)
    library = tmp_path / "repos"
    library.mkdir()
    _make_python_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    result = service.repos_index(str(library))
    search = service.repos_search_task("arregla pytest")

    assert result["repo_count"] == 1
    assert search["results"]


def test_repo_library_handles_missing_and_corrupt_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    service = CodeAgentRuntimeService(project)

    missing = service.repos_list()
    index_path = project / "runtime" / "code_agent" / "repo_library_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{not json", encoding="utf-8")
    corrupt = service.repos_stats()

    assert missing["repo_count"] == 0
    assert corrupt["repo_count"] == 0
    assert corrupt["warnings"]
    assert list(index_path.parent.glob("repo_library_index.json.corrupt-*.bak"))


def test_repo_library_skips_symlink_outside_library(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    outside = tmp_path / "outside"
    library.mkdir()
    outside.mkdir()
    _make_python_repo(outside)
    link = library / "linked-outside"
    try:
        os.symlink(outside / "python-agent", link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")
    service = CodeAgentRuntimeService(tmp_path / "project")

    result = service.repos_index(str(library))

    assert result["repo_count"] == 0


def test_repo_library_cli_index_list_search_show_stats(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_python_repo(library)
    project = tmp_path / "project"
    runner = CliRunner()

    index = runner.invoke(app, ["code", "repos", "index", "--root", str(project), "--library-root", str(library)])
    listed = runner.invoke(app, ["code", "repos", "list", "--root", str(project), "--limit", "5"])
    search = runner.invoke(app, ["code", "repos", "search", "pytest fixtures", "--root", str(project)])
    task = runner.invoke(app, ["code", "repos", "search-task", "arregla pytest", "--root", str(project)])
    show = runner.invoke(app, ["code", "repos", "show", "python-agent", "--root", str(project)])
    stats = runner.invoke(app, ["code", "repos", "stats", "--root", str(project)])

    assert index.exit_code == 0
    assert '"repo_count": 1' in index.stdout
    assert listed.exit_code == 0
    assert "python-agent" in listed.stdout
    assert search.exit_code == 0
    assert "reference only" in search.stdout
    assert task.exit_code == 0
    assert "suggested_skills" in task.stdout
    assert show.exit_code == 0
    assert '"snippet_count"' in show.stdout
    assert stats.exit_code == 0
    assert '"snippet_count"' in stats.stdout


def test_repo_library_records_memory_without_secrets(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_python_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    service.repos_search_task("audita token password .env")
    memory = service.memory_show()
    raw = str(memory).casefold()

    assert memory["repo_library_indexes"]
    assert memory["repo_library_searches"]
    assert "[redacted]" in raw
    assert "token password" not in raw
    assert ".env" not in raw
