"""Bounded discovery over the current executable ActionRegistry."""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from .models import ActionResult, ActionRisk
from .registry import ActionContext, ActionDefinition, ActionRegistry


DISCOVERY_ACTION = "capabilities.search"


class CapabilitySearchPayload(BaseModel):
    query: str = Field(min_length=2, max_length=240)
    family: str | None = Field(default=None, max_length=64)
    max_results: int = Field(default=5, ge=1, le=8)


def _terms(value: str) -> set[str]:
    folded = "".join(
        char for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )
    return set(re.findall(r"[^\W_]+", folded, flags=re.UNICODE))


def search_actions(
    registry: ActionRegistry, query: str, *, family: str | None = None,
    max_results: int = 5,
) -> list[dict[str, object]]:
    """Rank current definitions; never return executable schemas or invoke handlers."""
    query_terms = _terms(query)
    query_folded = query.casefold().strip()
    ranked: list[tuple[int, ActionDefinition]] = []
    for definition in registry.list_actions():
        if definition.name == DISCOVERY_ACTION:
            continue
        definition_family, _, operation = definition.name.partition(".")
        if family and definition_family.casefold() != family.casefold():
            continue
        name_terms = _terms(definition.name)
        description_terms = _terms(definition.description)
        alias_terms = set().union(*(_terms(alias) for alias in definition.semantic_aliases)) if definition.semantic_aliases else set()
        score = 8 * len(query_terms & name_terms)
        score += 3 * len(query_terms & alias_terms)
        score += min(len(query_terms & description_terms), 4)
        if query_folded == definition.name.casefold():
            score += 40
        if operation.casefold() in query_folded:
            score += 6
        if definition_family.casefold() in query_folded:
            score += 4
        if score > 0 or family:
            ranked.append((score, definition))
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return [
        {
            "name": definition.name,
            "family": definition.name.split(".", 1)[0],
            "operation": definition.name.rsplit(".", 1)[-1],
            "description": definition.description[:240],
            "risk": definition.risk.value,
            "available": True,
            "companions": list(definition.companion_names[:4]),
        }
        for _, definition in ranked[:max_results]
    ]


def register_discovery_action(registry: ActionRegistry) -> None:
    def discover(_context: ActionContext, payload: CapabilitySearchPayload) -> ActionResult:
        candidates = search_actions(
            registry, payload.query, family=payload.family,
            max_results=payload.max_results,
        )
        return ActionResult(
            message=f"Found {len(candidates)} current executable capabilities.",
            data={"candidates": candidates, "query": payload.query, "count": len(candidates)},
        )

    registry.register(ActionDefinition(
        name=DISCOVERY_ACTION,
        description=(
            "Search CURRENT executable JARVIS capabilities by task, family, or operation. "
            "Returns bounded summaries, not tool execution or permission. Use when a needed "
            "detailed schema is missing or uncertain; the next agent step can receive schemas "
            "for returned candidates. This is not Internet search."
        ),
        payload_model=CapabilitySearchPayload,
        handler=discover,
        risk=ActionRisk.SAFE,
        idempotent=True,
        semantic_aliases=("discover capability", "find available tool", "search tools", "buscar capacidad", "descubrir herramienta"),
    ))
