from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from jarvis.code_agent_runtime.paths import is_inside_project
from jarvis.code_agent_runtime.repo_library.repo_index import RepoLibraryIndex
from jarvis.code_agent_runtime.repo_learning.github_discovery import GitHubDiscovery
from jarvis.code_agent_runtime.repo_learning.learning_extractor import LearningExtractor
from jarvis.code_agent_runtime.repo_learning.models import LEARNING_NOTICE
from jarvis.code_agent_runtime.repo_learning.pattern_store import PatternStore


class RepoLearningRouter:
    def __init__(self, *, discovery: GitHubDiscovery, repo_library: RepoLibraryIndex, pattern_store: PatternStore, extractor: LearningExtractor | None = None) -> None:
        self._discovery = discovery
        self._repo_library = repo_library
        self._patterns = pattern_store
        self._extractor = extractor or LearningExtractor()
        self._last_search: dict[str, dict[str, Any]] = {}

    def search_github(self, query: str, *, max_results: int = 20, language: str | None = None, topic: str | None = None) -> dict[str, Any]:
        result = self._discovery.search(query, max_results=max_results, language=language, topic=topic)
        result["results"] = [self._score_candidate(item, query) for item in result.get("results", []) if isinstance(item, dict)]
        self._last_search = {item["id"]: item for item in result.get("results", []) if isinstance(item, dict)}
        result["notice"] = LEARNING_NOTICE
        return result

    def shortlist(self, task: str, *, max_results: int = 10, language: str | None = None, topic: str | None = None) -> dict[str, Any]:
        result = self.search_github(task, max_results=max_results, language=language, topic=topic)
        candidates = sorted(result.get("results", []), key=lambda item: item.get("learning_value", 0), reverse=True)
        return {
            "status": result.get("status"),
            "task": task,
            "candidates": candidates[:max_results],
            "message": "Review licenses and risks. Cloning requires explicit confirmation.",
            "notice": LEARNING_NOTICE,
        }

    def clone(self, repo_id: str, *, library_root: str | Path, confirm: bool = False, overwrite: bool = False) -> dict[str, Any]:
        if not confirm:
            return {"status": "confirmation_required", "message": "cloning public GitHub repos requires explicit confirmation", "repo_id": repo_id}
        candidate = self._candidate(repo_id)
        if candidate.get("private"):
            return {"status": "blocked", "message": "private repositories are not allowed", "repo_id": repo_id}
        if not candidate.get("clone_url", "").startswith("https://github.com/"):
            return {"status": "blocked", "message": "only public GitHub HTTPS clone URLs are allowed", "repo_id": repo_id}
        root = Path(library_root).expanduser().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        try:
            folder_name = self.sanitize_folder_name(candidate.get("full_name") or repo_id)
        except ValueError as exc:
            return {"status": "blocked", "message": str(exc), "repo_id": repo_id}
        target = (root / folder_name).resolve(strict=False)
        if not is_inside_project(root, target):
            return {"status": "blocked", "message": "target path resolves outside library root", "repo_id": repo_id}
        if target.exists():
            return {
                "status": "blocked" if overwrite else "confirmation_required",
                "message": "target folder already exists; refusing to overwrite existing repository folder",
                "target": str(target),
                "repo_id": repo_id,
            }
        command = ["git", "clone", "--depth", "1", "--", candidate["clone_url"], str(target)]
        try:
            completed = subprocess.run(command, cwd=str(root), shell=False, capture_output=True, text=True, timeout=180)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "message": str(exc), "repo_id": repo_id}
        status = "ok" if completed.returncode == 0 else "failed"
        return {
            "status": status,
            "repo_id": repo_id,
            "full_name": candidate.get("full_name"),
            "target": str(target),
            "return_code": completed.returncode,
            "stderr": completed.stderr[-1000:],
            "warnings": self._candidate_warnings(candidate),
            "notice": LEARNING_NOTICE,
        }

    def clone_and_index(self, repo_id: str, *, library_root: str | Path, confirm: bool = False, overwrite: bool = False) -> dict[str, Any]:
        clone_result = self.clone(repo_id, library_root=library_root, confirm=confirm, overwrite=overwrite)
        if clone_result.get("status") != "ok":
            return {"clone": clone_result, "indexed": None, "notice": LEARNING_NOTICE}
        indexed = self._repo_library.index(library_root)
        extracted = self.extract()
        return {"clone": clone_result, "indexed": indexed, "extracted": extracted, "notice": LEARNING_NOTICE}

    def extract(self) -> dict[str, Any]:
        entries = self._extractor.extract(self._repo_library.records())
        payload = self._patterns.replace(entries)
        return {"entry_count": payload.get("entry_count", len(entries)), "notice": LEARNING_NOTICE}

    def list(self, *, limit: int = 100) -> dict[str, Any]:
        return self._patterns.list(limit=limit)

    def search(self, query: str, *, skill_ids: list[str] | None = None, limit: int = 10) -> dict[str, Any]:
        return self._patterns.search(query, skill_ids=skill_ids, limit=limit)

    def for_task(self, task: str, *, skill_ids: list[str], limit: int = 8) -> dict[str, Any]:
        query = " ".join([task, *skill_ids])
        result = self.search(query, skill_ids=skill_ids, limit=limit)
        result["task"] = task
        return result

    def stats(self) -> dict[str, Any]:
        return self._patterns.stats()

    def summary(self, *, limit: int = 8) -> dict[str, Any]:
        stats = self.stats()
        entries = self.list(limit=limit).get("entries", [])
        return {
            "status": "ok",
            "entry_count": stats.get("entry_count", 0),
            "skills": stats.get("skills", []),
            "recent_entries": entries,
            "notice": LEARNING_NOTICE,
        }

    def _candidate(self, repo_id: str) -> dict[str, Any]:
        if repo_id in self._last_search:
            return self._last_search[repo_id]
        if "/" in repo_id:
            full_name = repo_id
        else:
            full_name = repo_id.replace("__", "/")
        return {
            "id": repo_id,
            "full_name": full_name,
            "clone_url": f"https://github.com/{full_name}.git",
            "private": False,
            "license": "unknown",
        }

    @staticmethod
    def _candidate_warnings(candidate: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        full_name = candidate.get("full_name") or candidate.get("id") or "unknown"
        if candidate.get("archived"):
            warnings.append(f"archived repository: {full_name}")
        if candidate.get("fork"):
            warnings.append(f"fork repository: {full_name}")
        if str(candidate.get("license") or "unknown").casefold() == "unknown":
            warnings.append(f"unknown license: {full_name}")
        return warnings

    @staticmethod
    def _score_candidate(candidate: dict[str, Any], query: str) -> dict[str, Any]:
        scored = dict(candidate)
        query_terms = {part for part in re.split(r"\W+", query.casefold()) if len(part) > 2}
        haystack = " ".join(
            [
                str(candidate.get("full_name") or ""),
                str(candidate.get("description") or ""),
                str(candidate.get("language") or ""),
                " ".join(str(item) for item in candidate.get("topics", []) if isinstance(item, str)),
            ]
        ).casefold()
        matches = sum(1 for term in query_terms if term in haystack)
        relevance = min(1.0, matches / max(1, min(len(query_terms), 8)))
        stars = int(candidate.get("stargazers_count") or 0)
        quality = min(1.0, 0.25 + (stars / 5000.0))
        if candidate.get("archived"):
            quality -= 0.35
        if candidate.get("fork"):
            quality -= 0.15
        license_value = str(candidate.get("license") or "unknown").casefold()
        license_risk = "unknown" if license_value == "unknown" else "low"
        security_risk = "high" if candidate.get("private") else ("medium" if candidate.get("archived") else "low")
        learning_value = max(0.0, round((relevance * 0.5) + (max(0.0, quality) * 0.35) + (0.15 if license_risk == "low" else 0.0), 3))
        scored.update(
            {
                "relevance_score": round(relevance, 3),
                "quality_score": round(max(0.0, quality), 3),
                "freshness_score": 0.5,
                "license_risk": license_risk,
                "security_risk": security_risk,
                "learning_value": learning_value,
                "warnings": RepoLearningRouter._candidate_warnings(candidate),
            }
        )
        return scored

    @staticmethod
    def sanitize_folder_name(value: str) -> str:
        folded = value.strip().casefold().replace("/", "__")
        if any(part in folded for part in ("..", "~", "^", ":", "\\", "@{")) or folded.startswith(("-", "/", ".")):
            raise ValueError(f"unsafe repository folder name: {value}")
        safe = re.sub(r"[^a-z0-9_.-]+", "-", folded).strip("-._")
        if not safe:
            raise ValueError("repository folder name cannot be empty")
        return safe[:120]
