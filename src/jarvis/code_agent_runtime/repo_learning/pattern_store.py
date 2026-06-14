from __future__ import annotations

import re
from typing import Any

from jarvis.code_agent_runtime.repo_learning.models import LEARNING_NOTICE, LearningEntry
from jarvis.code_agent_runtime.repo_learning.storage import LearningStorage


class PatternStore:
    def __init__(self, storage: LearningStorage) -> None:
        self._storage = storage

    def replace(self, entries: list[LearningEntry]) -> dict[str, Any]:
        return self._storage.save_entries(entries)

    def list(self, *, limit: int = 100) -> dict[str, Any]:
        entries = self._storage.entries()
        return {
            "entry_count": len(entries),
            "returned_count": min(limit, len(entries)),
            "omitted_count": max(0, len(entries) - limit),
            "entries": [entry.to_dict() for entry in entries[:limit]],
            "warnings": self._storage.load().get("warnings", []),
            "notice": LEARNING_NOTICE,
        }

    def stats(self) -> dict[str, Any]:
        entries = self._storage.entries()
        by_skill: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for entry in entries:
            by_skill[entry.skill] = by_skill.get(entry.skill, 0) + 1
            by_type[entry.learning_type] = by_type.get(entry.learning_type, 0) + 1
        return {
            "entry_count": len(entries),
            "by_skill": dict(sorted(by_skill.items())),
            "by_type": dict(sorted(by_type.items())),
            "storage_path": str(self._storage.path),
            "warnings": self._storage.load().get("warnings", []),
            "notice": LEARNING_NOTICE,
        }

    def search(self, query: str, *, skill_ids: list[str] | None = None, limit: int = 10) -> dict[str, Any]:
        terms = self._terms(query)
        skills = set(skill_ids or [])
        results: list[tuple[int, LearningEntry, str]] = []
        for entry in self._storage.entries():
            text = " ".join([entry.title, entry.summary, entry.observed_pattern, entry.when_to_use, entry.source_file, " ".join(entry.tags)]).casefold()
            score = 0
            matched: list[str] = []
            for term in terms:
                if term in text:
                    score += 4 if len(term) >= 5 else 2
                    matched.append(term)
            if skills and entry.skill in skills:
                score += 6
                matched.append(f"skill:{entry.skill}")
            if score > 0:
                results.append((score, entry, ", ".join(matched[:8]) or "skill match"))
        results.sort(key=lambda item: (-item[0], item[1].repo_name, item[1].source_file))
        return {
            "query": query,
            "skill_ids": sorted(skills),
            "result_count": min(limit, len(results)),
            "results": [
                entry.to_dict() | {"score": score, "match_reason": reason}
                for score, entry, reason in results[:limit]
            ],
            "notice": LEARNING_NOTICE,
        }

    @staticmethod
    def _terms(query: str) -> list[str]:
        return [term for term in re.split(r"[^A-Za-z0-9_#.+-]+", query.casefold()) if len(term) >= 2]
