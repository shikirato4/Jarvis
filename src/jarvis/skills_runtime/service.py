from __future__ import annotations

from .models import Skill, SkillCatalog
from .registry import SkillRegistry

_TRUNCATION_NOTE = "... [truncated]"


class SkillService:
    """Selects loaded skills by trigger relevance and renders prompt layers."""

    def __init__(
        self,
        catalog: SkillCatalog,
        *,
        max_prompt_skills: int = 2,
        max_body_chars: int = 1_200,
    ) -> None:
        self._catalog = catalog
        self._max_prompt_skills = max(1, max_prompt_skills)
        self._max_body_chars = max(200, max_body_chars)

    @property
    def catalog(self) -> SkillCatalog:
        return self._catalog

    def select_for_query(self, text: str) -> list[Skill]:
        """Rank skills by how many of their triggers the query mentions."""
        lowered = (text or "").casefold()
        if not lowered.strip():
            return []
        matches: list[tuple[int, str, Skill]] = []
        for skill in self._catalog.skills:
            hits = sum(
                1
                for trigger in skill.triggers
                if trigger.casefold() and trigger.casefold() in lowered
            )
            if hits > 0:
                matches.append((-hits, skill.name, skill))
        matches.sort()
        return [skill for _, _, skill in matches[: self._max_prompt_skills]]

    def skills_context(self, text: str) -> str:
        """Render the matching skills as a guidance block, empty if none match."""
        selected = self.select_for_query(text)
        if not selected:
            return ""
        rendered: list[str] = [
            "Domain procedures for this request. Follow them where they apply.",
        ]
        for skill in selected:
            body = skill.body
            if len(body) > self._max_body_chars:
                body = body[: self._max_body_chars] + _TRUNCATION_NOTE
            rendered.append(
                f"[{skill.name}] (v{skill.version}) — {skill.description}\n{body}"
            )
        return "\n\n".join(rendered)

    def status(self) -> dict[str, object]:
        return {
            "loaded": len(self._catalog.skills),
            "warnings_count": len(self._catalog.warnings),
            "warnings": list(self._catalog.warnings),
            "names": [skill.name for skill in self._catalog.skills],
        }


__all__ = ["SkillService", "SkillRegistry", "Skill", "SkillCatalog"]
