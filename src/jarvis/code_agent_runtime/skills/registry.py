from __future__ import annotations

from jarvis.code_agent_runtime.skills.base import CodeAgentSkill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, CodeAgentSkill] = {}

    def register(self, skill: CodeAgentSkill) -> None:
        if skill.id in self._skills:
            raise ValueError(f"duplicate skill id: {skill.id}")
        self._skills[skill.id] = skill

    def get(self, skill_id: str) -> CodeAgentSkill:
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"skill not found: {skill_id}") from exc

    def list(self) -> list[CodeAgentSkill]:
        return [self._skills[key] for key in sorted(self._skills)]

    def by_tag(self, tag: str) -> list[CodeAgentSkill]:
        return [skill for skill in self.list() if skill.matches_tag(tag)]

    def ids(self) -> list[str]:
        return sorted(self._skills)
