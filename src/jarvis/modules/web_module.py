from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.web_search import build_web_search_provider
from jarvis.web_search.providers import WebSearchProvider


class WebSearchPayload(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    max_results: int = Field(default=5, ge=1, le=10)


def build_web_search_action(provider: WebSearchProvider) -> ActionDefinition:
    def search(_context: ActionContext, payload: WebSearchPayload) -> ActionResult:
        response = provider.search(payload.query, max_results=payload.max_results)
        # A bounded provider failure is a valid tool observation. Returning it
        # to the model as data lets local and online agents continue normally
        # (and explain the limitation) without pretending that Brave worked.
        successful = response.status != "blocked"
        return ActionResult(
            status=(
                ExecutionStatus.SUCCESS if successful else ExecutionStatus.FAILED
            ),
            message=(
                "web search request handled"
                if successful
                else "web search request blocked"
            ),
            data=response.model_dump(mode="json"),
        )

    return ActionDefinition(
        name="web.search",
        description=(
            "Search the public web or Internet for current information (buscar, "
            "busca, noticias, novedades, fuentes). Use only when the user needs "
            "recent facts or external sources; never include secrets."
        ),
        payload_model=WebSearchPayload,
        handler=search,
        tags=("web", "search", "research"),
        idempotent=True,
        adapter="official_api",
        semantic_aliases=(
            "research current information", "investigar versión actual",
            "investiga", "busca fuentes", "buscar fuentes",
            "busca en internet", "novedades actuales",
            "search web", "latest release", "current version",
            "look up online", "consultar en web", "lanzamiento reciente",
        ),
    )


class WebModule:
    name = "web"
    description = "Public web search exposed as a provider-neutral JARVIS agent tool."

    def __init__(
        self,
        provider_factory: Callable[[], WebSearchProvider] = build_web_search_provider,
    ) -> None:
        self._provider_factory = provider_factory

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(build_web_search_action(self._provider_factory()))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="web.search",
                module_name=self.name,
                intent="web",
                description="Current public web information with source results.",
                action_names=("web.search",),
                keywords=("busca", "noticias", "fuentes", "web", "actual"),
                mode_policy=(
                    ExecutionMode.ASSIST,
                    ExecutionMode.RESEARCH,
                    ExecutionMode.OPERATOR,
                    ExecutionMode.AUTOMATION,
                ),
                task_type="research",
            )
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return
