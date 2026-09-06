from __future__ import annotations

from enum import StrEnum


class ReasoningPreference(StrEnum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    XHIGH = "xhigh"
    AUTO = "auto"


def resolve_reasoning_preference(
    preference: ReasoningPreference | str, prompt: str = ""
) -> ReasoningPreference:
    """Resolve Auto deterministically without coupling to an inference backend."""
    selected = ReasoningPreference(str(preference).casefold())
    if selected is not ReasoningPreference.AUTO:
        return selected
    lowered = prompt.casefold()
    if any(
        marker in lowered
        for marker in (
            "debug",
            "investiga",
            "research",
            "matem",
            "complejo",
            "multi-step",
            "agente",
        )
    ):
        return ReasoningPreference.XHIGH
    if len(prompt) > 500 or any(
        marker in lowered for marker in ("analiza", "explica", "programa", "código")
    ):
        return ReasoningPreference.MEDIUM
    return ReasoningPreference.LOW
