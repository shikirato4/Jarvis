from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .models import WebSearchHit, WebSearchProviderStatus, WebSearchResponse
from .sanitizer import sanitize_web_query

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_ENV_CACHE: dict[str, str] | None = None


class WebSearchProvider(Protocol):
    def status(self) -> WebSearchProviderStatus: ...

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse: ...


class DisabledWebSearchProvider:
    provider = "disabled"

    def status(self) -> WebSearchProviderStatus:
        return WebSearchProviderStatus(provider="disabled", message="web search disabled")

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return WebSearchResponse(status="disabled", provider=self.provider, query="", message="Web search is disabled.")


class BraveSearchProvider:
    provider = "brave"
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, *, api_key: str | None = None, enabled: bool | None = None, max_results: int | None = None, timeout_seconds: float | None = None) -> None:
        self._api_key = api_key if api_key is not None else _env_value("JARVIS_BRAVE_SEARCH_API_KEY", "")
        self._enabled = _env_bool("JARVIS_WEB_SEARCH_ENABLED", False) if enabled is None else enabled
        self._max_results = _bounded_int(max_results if max_results is not None else _env_value("JARVIS_WEB_SEARCH_MAX_RESULTS"), default=5, low=1, high=10)
        self._timeout_seconds = _bounded_float(timeout_seconds if timeout_seconds is not None else _env_value("JARVIS_WEB_SEARCH_TIMEOUT_SECONDS"), default=8.0, low=1.0, high=20.0)

    def status(self) -> WebSearchProviderStatus:
        configured = bool(self._api_key)
        return WebSearchProviderStatus(
            provider=self.provider,
            enabled=self._enabled,
            available=bool(self._enabled and configured),
            configured=configured,
            sends_private_context=False,
            max_results=self._max_results,
            timeout_seconds=self._timeout_seconds,
            message="ok" if self._enabled and configured else ("Brave Search API key not configured" if self._enabled else "web search disabled"),
        )

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        status = self.status()
        sanitized = sanitize_web_query(query)
        if not sanitized.allowed:
            return WebSearchResponse(status="blocked", provider=self.provider, query="", message=f"No puedo mandar esa informacion a internet: {sanitized.reason}. Puedo ayudarte en modo local/offline.")
        if not status.enabled:
            return WebSearchResponse(status="disabled", provider=self.provider, query=sanitized.query, message="Web search is disabled.")
        if not status.configured:
            return WebSearchResponse(status="unavailable", provider=self.provider, query=sanitized.query, message="Brave Search API key not configured.")

        limit = min(max(1, int(max_results or self._max_results)), self._max_results, 10)
        url = f"{self.API_URL}?{urlencode({'q': sanitized.query, 'count': limit})}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "Jarvis-WebSearch/1.0",
                "X-Subscription-Token": self._api_key,
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - fixed Brave API endpoint.
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            return WebSearchResponse(status="error", provider=self.provider, query=sanitized.query, message=_safe_error(f"Brave Search error HTTP {exc.code}"))
        except (URLError, TimeoutError, OSError) as exc:
            return WebSearchResponse(status="error", provider=self.provider, query=sanitized.query, message=_safe_error(f"Brave Search connection error: {exc}"))
        except Exception as exc:  # noqa: BLE001
            return WebSearchResponse(status="error", provider=self.provider, query=sanitized.query, message=_safe_error(f"Brave Search failed: {exc}"))

        hits = _parse_brave_hits(payload, limit)
        message = "ok" if hits else "No encontre resultados confiables."
        return WebSearchResponse(status="ok" if hits else "empty", provider=self.provider, query=sanitized.query, hits=hits, message=message)


def build_web_search_provider() -> WebSearchProvider:
    provider = _env_value("JARVIS_WEB_SEARCH_PROVIDER", "disabled").strip().casefold() or "disabled"
    enabled = _env_bool("JARVIS_WEB_SEARCH_ENABLED", False)
    if provider == "brave" or enabled:
        return BraveSearchProvider(enabled=enabled)
    return DisabledWebSearchProvider()


def _parse_brave_hits(payload: dict, limit: int) -> list[WebSearchHit]:
    results = payload.get("web", {}).get("results", [])
    hits: list[WebSearchHit] = []
    for index, item in enumerate(results[:limit], start=1):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not url or not title:
            continue
        parsed = urlparse(url)
        snippet = str(item.get("description") or item.get("snippet") or "").strip()
        hits.append(
            WebSearchHit(
                title=title[:180],
                url=url[:500],
                snippet=snippet[:420],
                source=(parsed.hostname or "").removeprefix("www.")[:120],
                rank=index,
                published_at=str(item.get("age") or item.get("page_age") or "")[:80] or None,
            )
        )
    return hits


def _safe_error(value: str) -> str:
    lowered = value.casefold()
    if any(term in lowered for term in ("token", "password", "secret", "credential", "api_key", "apikey", "key=")):
        return "Brave Search request failed."
    return value[:300]


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_value(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_value(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ.get(name, default)
    return _load_env_file().get(name, default)


def _load_env_file() -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    values: dict[str, str] = {}
    try:
        lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        _ENV_CACHE = values
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("JARVIS_"):
            continue
        values[key] = value.strip().strip('"').strip("'")
    _ENV_CACHE = values
    return values


def _bounded_int(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value) if value not in {None, ""} else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)


def _bounded_float(value: object, *, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value) if value not in {None, ""} else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, low), high)
