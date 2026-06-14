from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.code_agent_runtime.repo_library.models import REFERENCE_NOTICE, RepoRecord
from jarvis.code_agent_runtime.repo_library.repo_scanner import RepoLibraryScanner
from jarvis.code_agent_runtime.repo_library.repo_search import RepoLibrarySearch
from jarvis.code_agent_runtime.repo_library.storage import RepoLibraryStorage


class RepoLibraryIndex:
    def __init__(self, storage: RepoLibraryStorage, scanner: RepoLibraryScanner | None = None) -> None:
        self._storage = storage
        self._scanner = scanner or RepoLibraryScanner()

    @property
    def storage_path(self) -> Path:
        return self._storage.path

    def index(self, library_root: str | Path, *, max_repos: int | None = None) -> dict[str, Any]:
        root = Path(library_root).expanduser().resolve(strict=False)
        repos = self._scanner.scan_library(root, max_repos=max_repos)
        payload = self._storage.save(library_root=root, repos=repos)
        return self._summary(payload)

    def list(self, *, limit: int = 100) -> dict[str, Any]:
        payload = self._storage.load()
        repos = payload.get("repos", [])
        return {
            "library_root": payload.get("library_root", ""),
            "indexed_at": payload.get("indexed_at", ""),
            "repo_count": payload.get("repo_count", len(repos)),
            "returned_count": min(len(repos), limit),
            "omitted_count": max(0, len(repos) - limit),
            "repos": [RepoRecord.from_dict(item).to_dict(include_snippets=False) for item in repos[:limit]],
            "warnings": payload.get("warnings", []),
            "notice": REFERENCE_NOTICE,
        }

    def stats(self) -> dict[str, Any]:
        records = self._storage.records()
        languages: dict[str, int] = {}
        frameworks: dict[str, int] = {}
        snippet_count = 0
        for repo in records:
            snippet_count += len(repo.snippets)
            for language, count in repo.languages.items():
                languages[language] = languages.get(language, 0) + count
            for framework in repo.frameworks:
                frameworks[framework] = frameworks.get(framework, 0) + 1
        return {
            "repo_count": len(records),
            "snippet_count": snippet_count,
            "languages": dict(sorted(languages.items())),
            "frameworks": dict(sorted(frameworks.items())),
            "storage_path": str(self.storage_path),
            "warnings": self._storage.load().get("warnings", []),
            "notice": REFERENCE_NOTICE,
        }

    def records(self) -> list[RepoRecord]:
        return self._storage.records()

    def show(self, repo_id: str) -> dict[str, Any]:
        for repo in self._storage.records():
            if repo.id == repo_id:
                payload = repo.to_dict(include_snippets=False)
                payload["snippet_count"] = len(repo.snippets)
                payload["notice"] = REFERENCE_NOTICE
                return payload
        raise KeyError(f"repo not found: {repo_id}")

    def search(self, query: str, *, skill_ids: list[str] | None = None, limit: int = 10) -> dict[str, Any]:
        results = RepoLibrarySearch(self._storage.records()).search(query, skill_ids=skill_ids, limit=limit)
        return {
            "query": query,
            "skill_ids": skill_ids or [],
            "result_count": len(results),
            "results": [item.to_dict() for item in results],
            "notice": REFERENCE_NOTICE,
        }

    @staticmethod
    def _summary(payload: dict[str, Any]) -> dict[str, Any]:
        repos = payload.get("repos", [])
        return {
            "library_root": payload.get("library_root", ""),
            "indexed_at": payload.get("indexed_at", ""),
            "repo_count": len(repos),
            "snippet_count": sum(len(repo.get("snippets", [])) for repo in repos if isinstance(repo, dict)),
            "storage": "json",
            "notice": REFERENCE_NOTICE,
        }
