from __future__ import annotations

import re
from collections import Counter

from jarvis.config import Settings

from .models import FreshnessRequirement, ResearchDepth, WebSearchProviderStatus, WebSearchResponse
from .providers import (
    BraveSearchProvider,
    DisabledWebSearchProvider,
    ExaSearchProvider,
    SearXNGSearchProvider,
    WebSearchProvider,
    YouSearchProvider,
)


_TECHNICAL = re.compile(
    r"\b(paper|study|research|technical|documentation|docs|benchmark|model|modelo|"
    r"estudio|investigacion|investigación|cientific[oa]|arxiv|doi|github)\b",
    re.I,
)


class WebProviderRouter:
    """Select useful providers by intent; temporary failures remain retryable."""

    provider = "multi"

    def __init__(self, providers: list[WebSearchProvider]) -> None:
        self.providers = providers
        self.request_counts: Counter[str] = Counter()

    def statuses(self, *, force: bool = False) -> list[WebSearchProviderStatus]:
        return [item.check_health(force=force) if force else item.status() for item in self.providers]

    def status(self) -> WebSearchProviderStatus:
        statuses = self.statuses()
        configured = [item for item in statuses if item.configured and item.enabled]
        available = [item for item in configured if item.available]
        return WebSearchProviderStatus(
            provider=self.provider,
            enabled=bool(configured),
            configured=bool(configured),
            available=bool(available),
            reachable=None,
            working=None,
            configuration_state="configured" if configured else "missing_config",
            message=(
                "Configured Web providers: " + ", ".join(item.provider for item in configured)
                if configured
                else "No Web search provider is configured."
            ),
        )

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus:
        statuses = self.statuses(force=force)
        aggregate = self.status()
        working = [item.provider for item in statuses if item.working]
        return aggregate.model_copy(
            update={
                "working": bool(working),
                "available": bool(working),
                "message": "Working Web providers: " + ", ".join(working) if working else aggregate.message,
            }
        )

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return self.search_for(query, max_results=max_results, depth=ResearchDepth.NORMAL)

    def search_for(
        self,
        query: str,
        *,
        max_results: int,
        depth: ResearchDepth,
        freshness: FreshnessRequirement | None = None,
        missing_requirements: tuple[str, ...] = (),
    ) -> WebSearchResponse:
        del freshness, missing_requirements
        ordered = self._ordered(query)
        if not ordered:
            details = "; ".join(
                f"{status.provider}: {status.message}"
                for status in self.statuses()
                if status.provider != "disabled" and status.message
            )
            return WebSearchResponse(
                status="unavailable",
                provider=self.provider,
                query=query,
                error_code="missing_config",
                message=f"No Web search provider is configured. {details}".strip(),
            )
        target_successes = 2 if depth is ResearchDepth.DEEP and len(ordered) > 1 else 1
        hits = []
        errors = []
        used: list[str] = []
        successful = 0
        for provider in ordered:
            name = getattr(provider, "provider", "unknown")
            self.request_counts[name] += 1
            response = provider.search(query, max_results=max_results)
            used.append(name)
            if response.status == "ok":
                hits.extend(response.hits)
                successful += 1
                if successful >= target_successes:
                    break
            elif response.status != "empty":
                errors.append({"provider": name, "status": response.status, "error_code": response.error_code})
            if response.status == "empty" and successful + 1 >= target_successes:
                continue
        return WebSearchResponse(
            status="ok" if hits else ("empty" if not errors else "unavailable"),
            provider=self.provider,
            query=query,
            hits=hits,
            message="ok" if hits else "No configured provider returned results.",
            metadata={"providers_used": used, "provider_requests": dict(self.request_counts), "errors": errors},
        )

    def _ordered(self, query: str) -> list[WebSearchProvider]:
        available = [item for item in self.providers if item.status().enabled and item.status().configured]
        explicit = next((name for name in ("you", "brave", "exa", "searxng") if re.search(rf"\b{name}\b", query, re.I)), None)
        technical = bool(_TECHNICAL.search(query))
        priority = (
            [explicit, "you", "exa", "brave", "searxng"]
            if explicit
            else (["exa", "you", "brave", "searxng"] if technical else ["you", "brave", "exa", "searxng"])
        )
        order = {name: index for index, name in enumerate(dict.fromkeys(priority))}
        return sorted(available, key=lambda item: order.get(getattr(item, "provider", ""), 99))


def build_web_search_router(settings: Settings | None = None) -> WebProviderRouter:
    config = settings or Settings()
    if config.web_search_enabled is False:
        return WebProviderRouter([DisabledWebSearchProvider()])
    providers: list[WebSearchProvider] = [
        YouSearchProvider(settings=config),
        BraveSearchProvider(settings=config, enabled=True),
        ExaSearchProvider(settings=config),
        SearXNGSearchProvider(settings=config),
    ]
    return WebProviderRouter(providers)
