from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.search import SearchStorage


def _make_search_repo(root: Path) -> Path:
    repo = root / "python-agent-search"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("Python agent local search sqlite fts pytest fixtures", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='python-agent-search'\n", encoding="utf-8")
    (repo / "src" / "search_index.py").write_text("class SqliteSearchIndex:\n    def rebuild(self):\n        return 'fts bm25 query ranking'\n", encoding="utf-8")
    (repo / "src" / "security.py").write_text("def validate_command(command):\n    return 'command injection path traversal permissions'\n", encoding="utf-8")
    (repo / "tests" / "test_search.py").write_text("import pytest\n\ndef test_fixture():\n    assert True\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "noise.py").write_text("pytest secret", encoding="utf-8")
    return repo


def test_search_storage_detects_fts5_or_uses_fallback(tmp_path: Path) -> None:
    storage = SearchStorage(tmp_path / "search.sqlite")
    assert isinstance(storage.has_fts5(), bool)

    fallback = SearchStorage(tmp_path / "fallback.sqlite", force_fallback=True)
    result = fallback.rebuild([])

    assert result["backend"] == "text_fallback"
    assert result["warnings"]


def test_search_rebuild_indexes_repo_library_and_learning_without_secrets(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    service.learn_extract()
    rebuilt = service.local_search_rebuild()
    stats = service.local_search_stats()
    query = service.local_search_query("sqlite fts pytest fixtures", skill_ids=["python", "testing"], limit=5)
    raw = str(query).casefold()

    assert rebuilt["document_count"] >= 4
    assert stats["document_count"] == rebuilt["document_count"]
    assert query["results"]
    assert query["results"][0]["score"] > 0
    assert "reference" in query["notice"].casefold()
    assert ".env" not in raw
    assert "token=secret" not in raw
    assert all(len(item["snippet"]) <= 520 for item in query["results"])


def test_search_rebuild_creates_sqlite_fts5_index_when_available(tmp_path: Path) -> None:
    if not SearchStorage(tmp_path / "probe.sqlite").has_fts5():
        return
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    project = tmp_path / "project"
    service = CodeAgentRuntimeService(project)

    service.repos_index(str(library))
    rebuilt = service.local_search_rebuild()
    index_path = project / "runtime" / "code_agent" / "search_index.sqlite"

    assert rebuilt["backend"] == "sqlite_fts5"
    assert index_path.exists()
    with sqlite3.connect(index_path) as conn:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')")}
        rows = conn.execute("SELECT metadata FROM documents").fetchall()
    raw = str(rows).casefold()
    assert {"documents", "search_fts"}.issubset(table_names)
    assert ".env" not in raw
    assert "token=secret" not in raw


def test_search_task_uses_skills_and_ranks_security_patterns(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")
    service.repos_index(str(library))
    service.learn_extract()

    result = service.local_search_task("audita command injection path traversal", limit=5)

    assert {"security-audit", "testing"}.issubset({item["id"] for item in result["suggested_skills"]})
    assert result["results"]
    assert any("security-audit" in item["skills"] or "security-audit" in " ".join(item["match_reasons"]) for item in result["results"])


def test_search_context_for_task_is_limited_and_safe(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")
    service.repos_index(str(library))
    service.learn_extract()

    context = service.local_search_context_for_task("arregla errores de pytest con token password .env", max_results=5, max_chars=800)
    raw = str(context).casefold()

    assert len(context["context"]) <= 800
    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "token password" not in raw


def test_search_fallback_without_fts5(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(SearchStorage, "has_fts5", lambda self: False)
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    result = service.local_search_query("pytest fixture", skill_ids=["testing"])

    assert result["backend"] == "text_fallback"
    assert result["warnings"]
    assert result["results"]


def test_search_handles_missing_and_corrupt_index(tmp_path: Path) -> None:
    project = tmp_path / "project"
    service = CodeAgentRuntimeService(project)

    missing = service.local_search_stats()
    index = project / "runtime" / "code_agent" / "search_index.sqlite"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("not sqlite", encoding="utf-8")
    corrupt = service.local_search_query("pytest")

    assert missing["document_count"] == 0
    assert corrupt["warnings"]
    assert list(index.parent.glob("search_index.sqlite.corrupt-*.bak"))


def test_search_clear_confirm_only_removes_search_indexes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    runtime = project / "runtime" / "code_agent"
    runtime.mkdir(parents=True)
    sqlite_index = runtime / "search_index.sqlite"
    fallback_index = runtime / "search_index.json"
    memory = runtime / "project_memory.json"
    repo_index = runtime / "repo_library_index.json"
    sqlite_index.write_text("sqlite", encoding="utf-8")
    fallback_index.write_text("{}", encoding="utf-8")
    memory.write_text("{}", encoding="utf-8")
    repo_index.write_text("{}", encoding="utf-8")
    service = CodeAgentRuntimeService(project)

    blocked = service.local_search_clear()
    cleared = service.local_search_clear(confirm=True)

    assert blocked["status"] == "confirmation_required"
    assert sorted(Path(path).name for path in cleared["removed"]) == ["search_index.json", "search_index.sqlite"]
    assert not sqlite_index.exists()
    assert not fallback_index.exists()
    assert memory.exists()
    assert repo_index.exists()


def test_search_cli_rebuild_query_task_stats_clear(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    project = tmp_path / "project"
    runner = CliRunner()

    index = runner.invoke(app, ["code", "repos", "index", "--root", str(project), "--library-root", str(library)])
    extract = runner.invoke(app, ["code", "learn", "extract", "--root", str(project)])
    rebuild = runner.invoke(app, ["code", "search", "rebuild", "--root", str(project)])
    query = runner.invoke(app, ["code", "search", "query", "pytest fixture", "--root", str(project), "--skills", "python,testing"])
    task = runner.invoke(app, ["code", "search", "task", "arregla error de pytest", "--root", str(project)])
    context = runner.invoke(app, ["code", "search", "context", "arregla error de pytest", "--root", str(project), "--max-chars", "900"])
    stats = runner.invoke(app, ["code", "search", "stats", "--root", str(project)])
    project_search = runner.invoke(app, ["code", "search", "project", "missing", "--root", str(project)])
    clear_blocked = runner.invoke(app, ["code", "search", "clear", "--root", str(project)])
    clear = runner.invoke(app, ["code", "search", "clear", "--root", str(project), "--confirm"])

    assert index.exit_code == 0
    assert extract.exit_code == 0
    assert rebuild.exit_code == 0
    assert '"document_count"' in rebuild.stdout
    assert query.exit_code == 0
    assert "pytest" in query.stdout
    assert task.exit_code == 0
    assert "suggested_skills" in task.stdout
    assert context.exit_code == 0
    assert "context" in context.stdout
    assert stats.exit_code == 0
    assert project_search.exit_code == 0
    assert '"project_search"' in project_search.stdout
    assert clear_blocked.exit_code == 0
    assert "confirmation_required" in clear_blocked.stdout
    assert clear.exit_code == 0


def test_search_memory_records_events_without_snippets_or_secrets(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_search_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    service.local_search_query("token password .env pytest", skill_ids=["testing"])
    memory = service.memory_show()
    raw = str(memory).casefold()

    assert memory["local_search_events"]
    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "class sqlitesearchindex" not in raw
