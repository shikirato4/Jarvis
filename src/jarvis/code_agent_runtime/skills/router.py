from __future__ import annotations

from jarvis.code_agent_runtime.skills.base import SkillSuggestion
from jarvis.code_agent_runtime.skills.registry import SkillRegistry


class SkillRouter:
    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def suggest(self, task: str, project_summary: dict | None = None, *, limit: int = 5) -> list[SkillSuggestion]:
        suggestions: list[SkillSuggestion] = []
        for skill in self._registry.list():
            score = skill.score(task, project_summary)
            if score <= 0:
                continue
            suggestions.append(SkillSuggestion(skill=skill, score=score, reason=self._reason(task, skill.id, score)))
        suggestions.sort(key=lambda item: (-item.score, item.skill.id))
        return suggestions[:limit]

    @staticmethod
    def _reason(task: str, skill_id: str, score: int) -> str:
        return f"{skill_id} matched task keywords and project context with score {score}"
