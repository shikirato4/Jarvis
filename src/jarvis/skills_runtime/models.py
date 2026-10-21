from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Skill(BaseModel):
    """A domain procedure loaded from disk, injected into prompts on relevance."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    triggers: tuple[str, ...] = ()
    version: int = 1
    body: str


class SkillCatalog(BaseModel):
    """Immutable result of loading a skills directory."""

    model_config = ConfigDict(frozen=True)

    skills: tuple[Skill, ...] = ()
    warnings: tuple[str, ...] = ()
