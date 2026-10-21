from __future__ import annotations

import json
import os
import socket
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from jarvis.config import Settings

from .models import WebSearchHit, WebSearchProviderStatus, WebSearchResponse
from .sanitizer import sanitize_web_query

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_SUPPORTED_KEY_NAMES = (
    "BRAVE_SEARCH_API_KEY",
    "JARVIS_BRAVE_SEARCH_API_KEY",
    "BRAVE_API_KEY",
)


class WebSearchProvider(Protocol):
    def status(self) -> WebSearchProviderStatus: ...

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus: ...

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse: ...


class DisabledWebSearchProvider:
    provider = "disabled"

    def status(self) -> WebSearchProviderStatus:
        return WebSearchProviderStatus(
            provider=self.provider,
            configuration_state="disabled",
            message="Web search is disabled.",
        )

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus:
        return self.status()

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return WebSearchResponse(
            status="disabled",
            provider=self.provider,
            query="",
            error_code="disabled",
            message="Web search is disabled.",
        )


class _JsonSearchProvider:
    """Shared bounded HTTP behavior for optional JSON search providers."""

    provider = "unknown"
    role = "optional"

    def __init__(self, *, configured: bool, enabled: bool, timeout_seconds: float, max_results: int, health_ttl_seconds: float) -> None:
        self._configured = configured
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds
        self._max_results = max_results
        self._health_ttl_seconds = health_ttl_seconds
        self._health_cache: WebSearchProviderStatus | None = None
        self._health_checked_at = 0.0

    def status(self) -> WebSearchProviderStatus:
        if self._health_cache is not None:
            return self._health_cache
        return WebSearchProviderStatus(
            provider=self.provider,
            enabled=self._enabled,
            available=self._enabled and self._configured,
            configured=self._configured,
            configuration_state="configured" if self._configured else "missing_config",
            sends_private_context=False,
            max_results=self._max_results,
            timeout_seconds=self._timeout_seconds,
            message=(f"{self.provider} is configured; health has not been checked yet." if self._configured else f"{self.provider} is not configured."),
        )

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus:
        if not force and self._health_cache is not None and time.monotonic() - self._health_checked_at < self._health_ttl_seconds:
            return self._health_cache
        if not self._enabled or not self._configured:
            return self.status()
        response = self.search("OpenAI", max_results=1)
        reachable = response.status not in {"timeout", "network_error"}
        authenticated = False if response.http_status in {401, 403} else (True if reachable else None)
        working = response.status in {"ok", "empty"}
        self._health_cache = WebSearchProviderStatus(
            provider=self.provider,
            enabled=self._enabled,
            available=working,
            configured=self._configured,
            authenticated=authenticated,
            reachable=reachable,
            working=working,
            configuration_state="configured",
            error_code=response.error_code,
            http_status=response.http_status,
            checked_at=datetime.now(UTC),
            sends_private_context=False,
            max_results=self._max_results,
            timeout_seconds=self._timeout_seconds,
            message=response.message,
        )
        self._health_checked_at = time.monotonic()
        return self._health_cache

    def _preflight(self, query: str) -> tuple[str, WebSearchResponse | None]:
        sanitized = sanitize_web_query(query)
        if not sanitized.allowed:
            return "", WebSearchResponse(status="blocked", provider=self.provider, query="", error_code="sensitive_query", message="Private or secret-bearing data cannot be sent to a Web provider.")
        if not self._enabled:
            return sanitized.query, WebSearchResponse(status="disabled", provider=self.provider, query=sanitized.query, error_code="disabled", message=f"{self.provider} is disabled.")
        if not self._configured:
            return sanitized.query, WebSearchResponse(status="unavailable", provider=self.provider, query=sanitized.query, error_code="missing_config", message=f"{self.provider} is not configured.")
        return sanitized.query, None

    def _post_json(self, url: str, payload: dict, headers: dict[str, str], query: str) -> dict | WebSearchResponse:
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Jarvis-Research/2.0", **headers}, method="POST")
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - provider endpoint is fixed/configured server-side.
                raw = response.read()
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("response root is not an object")
            return parsed
        except HTTPError as exc:
            return _provider_http_error(self.provider, query, exc)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return WebSearchResponse(status="invalid_response", provider=self.provider, query=query, error_code="invalid_response", message=f"{self.provider} returned an invalid response.")
        except TimeoutError:
            return WebSearchResponse(status="timeout", provider=self.provider, query=query, error_code="timeout", message=f"{self.provider} timed out.")
        except (URLError, OSError):
            return WebSearchResponse(status="network_error", provider=self.provider, query=query, error_code="network_error", message=f"Could not reach {self.provider}.")


class YouSearchProvider(_JsonSearchProvider):
    provider = "you"
    role = "general_primary"
    API_URL = "https://ydc-index.io/v1/search"

    def __init__(self, *, api_key: str | None = None, settings: Settings | None = None, enabled: bool = True) -> None:
        config = settings or Settings()
        secret = config.you_search_api_key
        self._api_key = (api_key if api_key is not None else (secret.get_secret_value() if secret else "")).strip()
        super().__init__(configured=bool(self._api_key), enabled=enabled, timeout_seconds=config.web_search_timeout_seconds, max_results=min(config.search_max_results_per_query, 100), health_ttl_seconds=config.web_search_health_ttl_seconds)

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        query, blocked = self._preflight(query)
        if blocked is not None:
            return blocked
        payload = {"query": query, "count": min(max(1, max_results), 100), "extraction": {"extraction_mode": "highlights"}}
        parsed = self._post_json(self.API_URL, payload, {"X-API-Key": self._api_key}, query)
        if isinstance(parsed, WebSearchResponse):
            return parsed
        hits: list[WebSearchHit] = []
        result_root = parsed.get("results") or {}
        sections = []
        if isinstance(result_root, dict):
            sections = [result_root.get("web", []), result_root.get("news", [])]
        elif isinstance(result_root, list):
            sections = [result_root]
        for section in sections:
            for item in section if isinstance(section, list) else []:
                hit = _normalized_hit(item, self.provider, len(hits) + 1)
                if hit is not None:
                    hits.append(hit)
                if len(hits) >= min(max_results, 100):
                    break
        return WebSearchResponse(status="ok" if hits else "empty", provider=self.provider, query=query, hits=hits, message="ok" if hits else "You.com returned no results.", http_status=200)


class ExaSearchProvider(_JsonSearchProvider):
    provider = "exa"
    role = "semantic_technical"
    API_URL = "https://api.exa.ai/search"

    def __init__(self, *, api_key: str | None = None, settings: Settings | None = None, enabled: bool = True) -> None:
        config = settings or Settings()
        secret = config.exa_search_api_key
        self._api_key = (api_key if api_key is not None else (secret.get_secret_value() if secret else "")).strip()
        super().__init__(configured=bool(self._api_key), enabled=enabled, timeout_seconds=config.web_search_timeout_seconds, max_results=min(config.search_max_results_per_query, 100), health_ttl_seconds=config.web_search_health_ttl_seconds)

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        query, blocked = self._preflight(query)
        if blocked is not None:
            return blocked
        payload = {"query": query, "type": "auto", "numResults": min(max(1, max_results), 100), "contents": {"highlights": {"query": query, "maxCharacters": 1800}}}
        parsed = self._post_json(self.API_URL, payload, {"x-api-key": self._api_key}, query)
        if isinstance(parsed, WebSearchResponse):
            return parsed
        results = parsed.get("results", [])
        hits = [hit for index, item in enumerate(results if isinstance(results, list) else [], 1) if (hit := _normalized_hit(item, self.provider, index)) is not None]
        return WebSearchResponse(status="ok" if hits else "empty", provider=self.provider, query=query, hits=hits[:max_results], message="ok" if hits else "Exa returned no results.", http_status=200)


class SearXNGSearchProvider(_JsonSearchProvider):
    provider = "searxng"
    role = "optional_coverage"

    def __init__(self, *, base_url: str | None = None, settings: Settings | None = None, enabled: bool = True) -> None:
        config = settings or Settings()
        self._base_url = (base_url if base_url is not None else config.searxng_base_url or "").strip().rstrip("/")
        super().__init__(configured=bool(self._base_url), enabled=enabled, timeout_seconds=config.web_search_timeout_seconds, max_results=config.search_max_results_per_query, health_ttl_seconds=config.web_search_health_ttl_seconds)

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        query, blocked = self._preflight(query)
        if blocked is not None:
            return blocked
        url = f"{self._base_url}/search?{urlencode({'q': query, 'format': 'json', 'safesearch': 1})}"
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "Jarvis-Research/2.0"})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310 - explicit server-side SearXNG configuration.
                parsed = json.loads(response.read().decode("utf-8"))
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
        except HTTPError as exc:
            return _provider_http_error(self.provider, query, exc)
        except TimeoutError:
            return WebSearchResponse(status="timeout", provider=self.provider, query=query, error_code="timeout", message="SearXNG timed out.")
        except (URLError, OSError):
            return WebSearchResponse(status="network_error", provider=self.provider, query=query, error_code="network_error", message="SearXNG is offline or unreachable.")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, AttributeError):
            return WebSearchResponse(status="invalid_response", provider=self.provider, query=query, error_code="invalid_response", message="SearXNG returned invalid JSON.")
        hits = [hit for index, item in enumerate(results if isinstance(results, list) else [], 1) if (hit := _normalized_hit(item, self.provider, index)) is not None]
        return WebSearchResponse(status="ok" if hits else "empty", provider=self.provider, query=query, hits=hits[:max_results], message="ok" if hits else "SearXNG returned no results.", http_status=200)

class BraveSearchProvider:
    provider = "brave"
    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        enabled: bool | None = None,
        max_results: int | None = None,
        timeout_seconds: float | None = None,
        health_ttl_seconds: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        config = settings or Settings()
        configured_secret = config.brave_search_api_key
        configured_key = configured_secret.get_secret_value() if configured_secret else ""
        self._api_key = (api_key if api_key is not None else configured_key).strip()
        self._key_source, self._key_present, self._misnamed_key = _inspect_key_configuration()
        if api_key is not None:
            self._key_source = "explicit"
            self._key_present = True
            self._misnamed_key = None
        configured_enabled = config.web_search_enabled
        self._enabled = (
            True if configured_enabled is None else configured_enabled
        ) if enabled is None else enabled
        self._max_results = _bounded_int(
            max_results if max_results is not None else config.web_search_max_results,
            default=5,
            low=1,
            high=10,
        )
        self._timeout_seconds = _bounded_float(
            timeout_seconds if timeout_seconds is not None else config.web_search_timeout_seconds,
            default=8.0,
            low=1.0,
            high=20.0,
        )
        self._health_ttl_seconds = _bounded_float(
            health_ttl_seconds if health_ttl_seconds is not None else config.web_search_health_ttl_seconds,
            default=300.0,
            low=0.0,
            high=3600.0,
        )
        self._health_cache: WebSearchProviderStatus | None = None
        self._health_checked_at = 0.0

    def status(self) -> WebSearchProviderStatus:
        if self._health_cache is not None:
            return self._health_cache
        configured = bool(self._api_key)
        configuration_state, message = self._configuration_state(configured)
        return WebSearchProviderStatus(
            provider=self.provider,
            enabled=self._enabled,
            available=bool(self._enabled and configured),
            configured=configured,
            configuration_state=configuration_state,
            key_source=self._key_source,
            env_file_found=_ENV_FILE.is_file(),
            sends_private_context=False,
            max_results=self._max_results,
            timeout_seconds=self._timeout_seconds,
            message=message,
        )

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus:
        if (
            not force
            and self._health_cache is not None
            and time.monotonic() - self._health_checked_at < self._health_ttl_seconds
        ):
            return self._health_cache
        baseline = self.status()
        if not baseline.enabled or not baseline.configured:
            return baseline
        result = self._request("OpenAI", 1)
        self._record_health(result)
        return self._health_cache or baseline

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        sanitized = sanitize_web_query(query)
        if not sanitized.allowed:
            return WebSearchResponse(
                status="blocked",
                provider=self.provider,
                query="",
                error_code="sensitive_query",
                message=f"No puedo mandar esa informacion a internet: {sanitized.reason}. Puedo ayudarte en modo local/offline.",
            )
        status = self.status()
        if not status.enabled:
            return WebSearchResponse(
                status="disabled",
                provider=self.provider,
                query=sanitized.query,
                error_code="disabled",
                message="Web search is disabled.",
            )
        if not status.configured:
            return WebSearchResponse(
                status="unavailable",
                provider=self.provider,
                query=sanitized.query,
                error_code=status.configuration_state,
                message=status.message,
            )

        limit = min(max(1, int(max_results or self._max_results)), self._max_results, 10)
        result = self._request(sanitized.query, limit)
        self._record_health(result)
        return result

    def _request(self, query: str, limit: int) -> WebSearchResponse:
        url = f"{self.API_URL}?{urlencode({'q': query, 'count': limit})}"
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
                raw_payload = response.read()
            payload = json.loads(raw_payload.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("response root is not an object")
            results = payload.get("web", {}).get("results", [])
            if not isinstance(results, list):
                raise ValueError("web.results is not a list")
        except HTTPError as exc:
            return _http_error_response(query, exc)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return WebSearchResponse(
                status="invalid_response",
                provider=self.provider,
                query=query,
                error_code="invalid_response",
                message="Brave Search returned an invalid response.",
            )
        except TimeoutError:
            return WebSearchResponse(
                status="timeout",
                provider=self.provider,
                query=query,
                error_code="timeout",
                message="Brave Search timed out. Try again shortly.",
            )
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return WebSearchResponse(
                    status="timeout",
                    provider=self.provider,
                    query=query,
                    error_code="timeout",
                    message="Brave Search timed out. Try again shortly.",
                )
            return WebSearchResponse(
                status="network_error",
                provider=self.provider,
                query=query,
                error_code="network_error",
                message="Could not reach Brave Search. Check the network connection.",
            )
        except OSError:
            return WebSearchResponse(
                status="network_error",
                provider=self.provider,
                query=query,
                error_code="network_error",
                message="Could not reach Brave Search. Check the network connection.",
            )
        except Exception:  # noqa: BLE001 - provider boundary must fail closed.
            return WebSearchResponse(
                status="internal_error",
                provider=self.provider,
                query=query,
                error_code="internal_error",
                message="Brave Search failed unexpectedly.",
            )

        hits = _parse_brave_hits(payload, limit)
        message = "ok" if hits else "Brave Search completed but returned no results."
        return WebSearchResponse(
            status="ok" if hits else "empty",
            provider=self.provider,
            query=query,
            hits=hits,
            message=message,
            http_status=200,
        )

    def _record_health(self, response: WebSearchResponse) -> None:
        reachable = response.status not in {"network_error", "timeout"}
        authenticated = None
        if response.http_status in {401, 403}:
            authenticated = False
        elif reachable and response.status != "invalid_response":
            authenticated = True
        working = response.status in {"ok", "empty"}
        self._health_cache = WebSearchProviderStatus(
            provider=self.provider,
            enabled=self._enabled,
            available=working,
            configured=bool(self._api_key),
            authenticated=authenticated,
            reachable=reachable,
            working=working,
            configuration_state="configured",
            key_source=self._key_source,
            env_file_found=_ENV_FILE.is_file(),
            error_code=response.error_code,
            http_status=response.http_status,
            checked_at=datetime.now(UTC),
            sends_private_context=False,
            max_results=self._max_results,
            timeout_seconds=self._timeout_seconds,
            message=response.message,
        )
        self._health_checked_at = time.monotonic()

    def _configuration_state(self, configured: bool) -> tuple[str, str]:
        if not self._enabled:
            return "disabled", "Web search is disabled."
        if configured:
            return "configured", "Brave Search is configured; health has not been checked yet."
        if self._misnamed_key:
            return "misnamed_variable", f"Unsupported Brave key variable '{self._misnamed_key}'. Use BRAVE_SEARCH_API_KEY."
        if self._key_present:
            return "empty_key", "BRAVE_SEARCH_API_KEY is present but empty."
        if not _ENV_FILE.is_file():
            return "env_file_missing", f"No .env file found at {_ENV_FILE}."
        return "missing_key", "BRAVE_SEARCH_API_KEY is not configured."


def build_web_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    config = settings or Settings()
    secret = config.brave_search_api_key
    has_key = bool(secret and secret.get_secret_value().strip())
    provider = (config.web_search_provider or "disabled").strip().casefold()
    explicitly_enabled = config.web_search_enabled
    enabled = (provider == "brave" or has_key) if explicitly_enabled is None else explicitly_enabled
    if provider == "brave" or enabled or has_key:
        return BraveSearchProvider(settings=config, enabled=enabled)
    return DisabledWebSearchProvider()


def check_brave_search_health(
    settings: Settings | None = None, *, force: bool = False
) -> WebSearchProviderStatus:
    """Run the bounded Brave probe on demand; normal status checks stay offline."""
    return build_web_search_provider(settings).check_health(force=force)


def _http_error_response(query: str, exc: HTTPError) -> WebSearchResponse:
    code = int(exc.code)
    retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
    if code == 401:
        status, error_code, message = "unauthorized", "authentication_failed", "Brave Search rejected the API key (HTTP 401)."
    elif code == 403:
        status, error_code, message = "forbidden", "forbidden", "Brave Search denied this request (HTTP 403). Check the plan and API permissions."
    elif code == 429:
        status, error_code, message = "rate_limited", "rate_limited", "Brave Search rate limit reached. Try again later."
    elif 500 <= code <= 599:
        status, error_code, message = "server_error", "server_error", f"Brave Search is temporarily unavailable (HTTP {code})."
    else:
        status, error_code, message = "http_error", "http_error", f"Brave Search request failed (HTTP {code})."
    return WebSearchResponse(
        status=status,
        provider="brave",
        query=query,
        error_code=error_code,
        http_status=code,
        retry_after_seconds=retry_after,
        message=message,
    )


def _provider_http_error(provider: str, query: str, exc: HTTPError) -> WebSearchResponse:
    code = int(exc.code)
    retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
    if code == 401:
        status, error_code = "unauthorized", "authentication_failed"
    elif code == 403:
        status, error_code = "forbidden", "forbidden"
    elif code == 429:
        status, error_code = "rate_limited", "rate_limited"
    elif 500 <= code <= 599:
        status, error_code = "server_error", "server_error"
    else:
        status, error_code = "http_error", "http_error"
    return WebSearchResponse(
        status=status,
        provider=provider,
        query=query,
        error_code=error_code,
        http_status=code,
        retry_after_seconds=retry_after,
        message=f"{provider} request failed (HTTP {code}).",
    )


def _normalized_hit(item: object, provider: str, rank: int) -> WebSearchHit | None:
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or item.get("link") or "").strip()
    title = str(item.get("title") or item.get("name") or "").strip()
    if not url or not title:
        return None
    raw_highlights = item.get("highlights") or []
    highlights = [str(value).strip() for value in raw_highlights if str(value).strip()] if isinstance(raw_highlights, list) else []
    raw_snippets = item.get("snippets") or []
    snippets: list[str] = []
    if isinstance(raw_snippets, list):
        for value in raw_snippets:
            text = value.get("snippet") if isinstance(value, dict) else value
            if text:
                snippets.append(str(text).strip())
    content = str(item.get("text") or item.get("content") or "").strip() or None
    snippet = str(item.get("description") or item.get("snippet") or "").strip()
    if not snippet:
        snippet = " ".join(highlights or snippets)[:1800]
    parsed = urlparse(url)
    return WebSearchHit(
        title=title[:300],
        url=url[:1200],
        snippet=snippet[:1800],
        source=(parsed.hostname or "").removeprefix("www.")[:160],
        provider=provider,
        rank=rank,
        published_at=str(item.get("publishedDate") or item.get("published_at") or item.get("publishedDateTime") or "")[:100] or None,
        author=str(item.get("author") or "")[:200] or None,
        highlights=highlights[:8],
        content=content[:10_000] if content else None,
        metadata={"provider_result_id": str(item.get("id") or "")[:500]},
    )


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


def _inspect_key_configuration() -> tuple[str | None, bool, str | None]:
    env_values = dict(os.environ)
    file_values = _read_env_file(_ENV_FILE)
    for name in _SUPPORTED_KEY_NAMES:
        if name in env_values:
            return f"environment:{name}", True, None
    for name in _SUPPORTED_KEY_NAMES:
        if name in file_values:
            return f".env:{name}", True, None
    all_names = set(env_values) | set(file_values)
    misnamed = next(
        (name for name in sorted(all_names) if "BRAVE" in name.upper() and "KEY" in name.upper()),
        None,
    )
    return None, False, misnamed


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _retry_after_seconds(value: str | None) -> int | None:
    try:
        return max(0, int(value)) if value else None
    except ValueError:
        return None


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
