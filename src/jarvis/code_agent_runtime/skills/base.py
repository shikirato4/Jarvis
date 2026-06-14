from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CodeAgentSkill:
    id: str
    name: str
    description: str
    tags: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    file_patterns: tuple[str, ...] = ()
    safe_commands: tuple[str, ...] = ()
    checklist: tuple[str, ...] = ()
    avoid: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def matches_tag(self, tag: str) -> bool:
        return tag.casefold() in {item.casefold() for item in self.tags}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "task_types": list(self.task_types),
            "file_patterns": list(self.file_patterns),
            "safe_commands": list(self.safe_commands),
            "checklist": list(self.checklist),
            "avoid": list(self.avoid),
        }

    def get_context(self, task: str, project_summary: dict | None = None) -> dict[str, Any]:
        summary = project_summary or {}
        key_files = summary.get("key_files") or summary.get("important_files") or []
        return {
            "skill_id": self.id,
            "recommended_focus": self.description,
            "task": task,
            "files_or_patterns_to_review": self._merge_patterns(key_files),
            "safe_commands_suggested": list(self.safe_commands),
            "checks": list(self.checklist),
            "risks_to_avoid": list(self.avoid),
        }

    def score(self, task: str, project_summary: dict | None = None) -> int:
        haystack = task.casefold()
        score = 0
        for keyword in self.keywords:
            if keyword.casefold() in haystack:
                score += 4
        for tag in self.tags:
            if tag.casefold() in haystack:
                score += 2
        for task_type in self.task_types:
            if task_type.casefold() in haystack:
                score += 2
        if score <= 0:
            return 0
        summary = project_summary or {}
        files = " ".join(str(item) for item in summary.get("key_files", []) + summary.get("important_files", []))
        for pattern in self.file_patterns:
            if pattern == "*":
                continue
            suffix = pattern.replace("*", "")
            if suffix and suffix in files:
                score += 1
        return score

    def _merge_patterns(self, key_files: list[str]) -> list[str]:
        values = [*self.file_patterns]
        for path in key_files[:10]:
            if any(path.endswith(pattern.replace("*", "")) for pattern in self.file_patterns if pattern.startswith("*")):
                values.append(path)
        return list(dict.fromkeys(values))[:16]


@dataclass(frozen=True)
class SkillSuggestion:
    skill: CodeAgentSkill
    score: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.skill.to_dict()
        payload["score"] = self.score
        payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class SkillContextBundle:
    task: str
    suggested_skills: tuple[SkillSuggestion, ...]
    memory_summary: str = ""
    git_summary: dict[str, Any] = field(default_factory=dict)
    skill_contexts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "suggested_skills": [item.to_dict() for item in self.suggested_skills],
            "memory_summary": self.memory_summary,
            "git_summary": self.git_summary,
            "skill_contexts": list(self.skill_contexts),
        }
