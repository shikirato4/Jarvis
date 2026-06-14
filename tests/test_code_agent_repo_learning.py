from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.repo_learning.learning_router import RepoLearningRouter


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _make_learning_repo(root: Path) -> Path:
    repo = root / "agent-memory"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("Agent memory tools permissions architecture", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='agent-memory'\n", encoding="utf-8")
    (repo / "src" / "memory.py").write_text("class JsonMemoryStore:\n    def recover_corrupt_json(self):\n        return 'safe memory pattern'\n", encoding="utf-8")
    (repo / "src" / "permissions.py").write_text("def validate_path(path):\n    return 'path traversal command injection security auth'\n", encoding="utf-8")
    (repo / "tests" / "test_memory.py").write_text("import pytest\n\ndef test_recovery():\n    assert True\n", encoding="utf-8")
    (repo / ".env").write_text("TOKEN=secret", encoding="utf-8")
    return repo


def test_github_search_uses_public_api_and_filters_private(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "items": [
            {
                "full_name": "public/agent",
                "clone_url": "https://github.com/public/agent.git",
                "html_url": "https://github.com/public/agent",
                "description": "agent memory",
                "language": "Python",
                "stargazers_count": 42,
                "archived": False,
                "private": False,
                "license": {"spdx_id": "MIT"},
                "topics": ["agent", "memory"],
            },
            {"full_name": "private/agent", "private": True},
        ]
    }

    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", lambda request, timeout=12: _FakeResponse(payload))
    result = CodeAgentRuntimeService(tmp_path).learn_search_github("python agent memory", max_results=10)

    assert result["status"] == "ok"
    assert result["result_count"] == 1
    assert result["results"][0]["id"] == "public__agent"
    assert result["results"][0]["license"] == "MIT"


def test_github_search_handles_rate_limit_and_offline(monkeypatch, tmp_path: Path) -> None:
    def rate_limited(request, timeout=12):  # noqa: ANN001
        raise HTTPError("https://api.github.com", 403, "rate limit", {}, None)

    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", rate_limited)
    limited = CodeAgentRuntimeService(tmp_path).learn_search_github("agent")

    def offline(request, timeout=12):  # noqa: ANN001
        raise URLError("offline")

    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", offline)
    offline_result = CodeAgentRuntimeService(tmp_path).learn_search_github("agent")

    assert limited["status"] == "rate_limited"
    assert limited["results"] == []
    assert offline_result["status"] == "offline"


def test_github_search_warns_for_archived_forks_unknown_license_and_empty(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "items": [
            {
                "full_name": "public/old-fork",
                "clone_url": "https://github.com/public/old-fork.git",
                "html_url": "https://github.com/public/old-fork",
                "archived": True,
                "fork": True,
                "private": False,
                "license": None,
            }
        ]
    }

    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", lambda request, timeout=12: _FakeResponse(payload))
    result = CodeAgentRuntimeService(tmp_path).learn_search_github("agent", max_results=10)

    assert result["status"] == "ok"
    assert result["results"][0]["fork"] is True
    assert any("archived repository" in warning for warning in result["warnings"])
    assert any("fork repository" in warning for warning in result["warnings"])
    assert any("unknown license" in warning for warning in result["warnings"])

    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", lambda request, timeout=12: _FakeResponse({"items": []}))
    empty = CodeAgentRuntimeService(tmp_path).learn_search_github("missing", max_results=10)

    assert empty["status"] == "ok"
    assert empty["result_count"] == 0
    assert "no public repositories found" in empty["message"]


def test_github_shortlist_scores_candidates_and_requires_review(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "items": [
            {
                "full_name": "public/pyside-chat",
                "clone_url": "https://github.com/public/pyside-chat.git",
                "html_url": "https://github.com/public/pyside-chat",
                "description": "PySide6 chat UI with tests",
                "language": "Python",
                "stargazers_count": 420,
                "archived": False,
                "fork": False,
                "private": False,
                "license": {"spdx_id": "MIT"},
                "topics": ["pyside6", "chat-ui"],
            },
            {
                "full_name": "public/old",
                "clone_url": "https://github.com/public/old.git",
                "html_url": "https://github.com/public/old",
                "description": "old example",
                "language": "Python",
                "stargazers_count": 1,
                "archived": True,
                "fork": True,
                "private": False,
                "license": None,
                "topics": [],
            },
        ]
    }
    monkeypatch.setattr("jarvis.code_agent_runtime.repo_learning.github_discovery.urlopen", lambda request, timeout=12: _FakeResponse(payload))

    result = CodeAgentRuntimeService(tmp_path).learn_shortlist("PySide6 chat UI", max_results=2)

    assert result["status"] == "ok"
    assert result["candidates"][0]["full_name"] == "public/pyside-chat"
    assert result["candidates"][0]["learning_value"] >= result["candidates"][1]["learning_value"]
    assert result["candidates"][1]["license_risk"] == "unknown"
    assert "confirmation" in result["message"].casefold()


def test_clone_requires_confirmation_and_sanitizes_paths(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        target = Path(command[-1])
        target.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = CodeAgentRuntimeService(tmp_path / "project")

    blocked = service.learn_clone("owner/repo", library_root=str(tmp_path / "repos"))
    ok = service.learn_clone("owner/repo", library_root=str(tmp_path / "repos"), confirm=True)

    assert blocked["status"] == "confirmation_required"
    assert calls == [calls[0]]
    assert ok["status"] == "ok"
    assert calls[0][:4] == ["git", "clone", "--depth", "1"]
    assert "--" in calls[0]
    assert Path(ok["target"]).name == "owner__repo"
    invalid = service.learn_clone("../repo", library_root=str(tmp_path / "repos"), confirm=True)
    assert invalid["status"] == "blocked"
    with pytest.raises(ValueError):
        RepoLearningRouter.sanitize_folder_name("../repo")
    with pytest.raises(ValueError):
        RepoLearningRouter.sanitize_folder_name("C:\\Users\\bad")


def test_clone_refuses_existing_target_even_with_overwrite(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    library = tmp_path / "repos"
    (library / "owner__repo").mkdir(parents=True)
    service = CodeAgentRuntimeService(tmp_path / "project")

    needs_confirmation = service.learn_clone("owner/repo", library_root=str(library), confirm=True)
    blocked = service.learn_clone("owner/repo", library_root=str(library), confirm=True, overwrite=True)

    assert needs_confirmation["status"] == "confirmation_required"
    assert blocked["status"] == "blocked"
    assert calls == []


def test_clone_blocks_private_or_non_github_candidates(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path / "project")
    service.executor.repo_learning._last_search = {
        "private": {"id": "private", "full_name": "x/y", "clone_url": "https://github.com/x/y.git", "private": True},
        "bad": {"id": "bad", "full_name": "x/y", "clone_url": "https://example.com/x/y.git", "private": False},
    }

    private = service.learn_clone("private", library_root=str(tmp_path / "repos"), confirm=True)
    bad = service.learn_clone("bad", library_root=str(tmp_path / "repos"), confirm=True)

    assert private["status"] == "blocked"
    assert bad["status"] == "blocked"


def test_clone_and_index_reuses_repo_library_after_controlled_clone(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        commands.append(command)
        repo = Path(command[-1])
        repo.mkdir(parents=True)
        (repo / "README.md").write_text("Python agent memory tools", encoding="utf-8")
        (repo / "memory.py").write_text("class MemoryStore:\n    pass\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    service = CodeAgentRuntimeService(tmp_path / "project")

    result = service.learn_clone_and_index("owner/repo", library_root=str(tmp_path / "repos"), confirm=True)

    assert result["clone"]["status"] == "ok"
    assert result["indexed"]["repo_count"] == 1
    assert result["extracted"]["entry_count"] >= 1
    assert commands == [["git", "clone", "--depth", "1", "--", "https://github.com/owner/repo.git", result["clone"]["target"]]]


def test_learning_extracts_patterns_from_repo_library(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_learning_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")
    service.repos_index(str(library))

    extracted = service.learn_extract()
    listed = service.learn_list()
    security = service.learn_search("path traversal command injection", skill_ids=["security-audit"])
    task = service.learn_for_task("haz memoria persistente con permisos")

    assert extracted["entry_count"] >= 3
    assert listed["entry_count"] == extracted["entry_count"]
    assert any(item["skill"] == "security-audit" for item in security["results"])
    assert task["results"]
    assert "suggested_skills" in task
    assert {"python", "debugging"}.intersection({item["id"] for item in task["suggested_skills"]})
    assert all(len(item["snippet"]) <= 700 for item in listed["entries"])


def test_learning_ignores_secrets_and_memory_does_not_store_snippets(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_learning_repo(library)
    service = CodeAgentRuntimeService(tmp_path / "project")

    service.repos_index(str(library))
    service.learn_extract()
    service.learn_for_task("audita token password .env")
    knowledge = service.learn_list()
    memory = service.memory_show()
    memory_raw = str(memory).casefold()
    knowledge_raw = str(knowledge).casefold()

    assert "[redacted]" in memory_raw
    assert ".env" not in knowledge_raw
    assert "token=secret" not in knowledge_raw
    assert "class jsonmemorystore" not in memory_raw


def test_learning_storage_handles_missing_and_corrupt_knowledge(tmp_path: Path) -> None:
    project = tmp_path / "project"
    service = CodeAgentRuntimeService(project)
    missing = service.learn_stats()
    path = project / "runtime" / "code_agent" / "repo_learning_knowledge.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")

    corrupt = service.learn_stats()

    assert missing["entry_count"] == 0
    assert corrupt["entry_count"] == 0
    assert corrupt["warnings"]
    assert list(path.parent.glob("repo_learning_knowledge.json.corrupt-*.bak"))


def test_learning_cli_basic_commands(tmp_path: Path) -> None:
    library = tmp_path / "repos"
    library.mkdir()
    _make_learning_repo(library)
    project = tmp_path / "project"
    runner = CliRunner()

    index = runner.invoke(app, ["code", "repos", "index", "--root", str(project), "--library-root", str(library)])
    extract = runner.invoke(app, ["code", "learn", "extract", "--root", str(project)])
    listed = runner.invoke(app, ["code", "learn", "list", "--root", str(project)])
    search = runner.invoke(app, ["code", "learn", "search", "memory permissions", "--root", str(project)])
    task = runner.invoke(app, ["code", "learn", "for-task", "haz memoria persistente", "--root", str(project)])
    stats = runner.invoke(app, ["code", "learn", "stats", "--root", str(project)])
    clone_blocked = runner.invoke(app, ["code", "learn", "clone", "owner/repo", "--root", str(project), "--library-root", str(library)])

    assert index.exit_code == 0
    assert extract.exit_code == 0
    assert '"entry_count"' in extract.stdout
    assert listed.exit_code == 0
    assert "reference patterns" in listed.stdout
    assert search.exit_code == 0
    assert "memory" in search.stdout
    assert task.exit_code == 0
    assert "suggested_skills" in task.stdout
    assert stats.exit_code == 0
    assert clone_blocked.exit_code == 0
    assert "confirmation_required" in clone_blocked.stdout
