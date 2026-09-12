from __future__ import annotations

import json
import socket
from urllib.error import HTTPError, URLError

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.config import Settings
from jarvis.web_search import BraveSearchProvider, WebSearchHit, WebSearchResponse, build_grounded_web_prompt, build_web_search_provider, sanitize_web_query, should_use_web_search
from jarvis.web_search.router import source_count_message


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _RawResponse(_FakeResponse):
    def __init__(self, payload: bytes) -> None:
        self._raw_payload = payload

    def read(self) -> bytes:
        return self._raw_payload


def test_sanitize_web_query_blocks_secrets_and_private_paths() -> None:
    blocked = [
        "busca OPENAI_API_KEY=sk-secret-value",
        "lee mi .env y busca TOKEN=abc",
        "-----BEGIN PRIVATE KEY----- abc",
        r"busca C:\Users\GAMER\Documents\jarvis\.env",
    ]

    for query in blocked:
        result = sanitize_web_query(query)
        assert result.allowed is False


def test_sanitize_web_query_allows_current_public_queries() -> None:
    result = sanitize_web_query("noticias de tecnologia hoy")

    assert result.allowed is True
    assert result.query == "noticias de tecnologia hoy"


def test_should_use_web_search_by_mode_and_task() -> None:
    assert should_use_web_search("noticias de tecnologia hoy", mode="auto") is True
    assert should_use_web_search("explica que es una lista enlazada", mode="auto") is False
    assert should_use_web_search("noticias de tecnologia hoy", mode="offline") is False
    assert should_use_web_search("hola", mode="online") is True
    assert should_use_web_search("edita este proyecto", mode="online") is False


def test_brave_status_never_exposes_api_key(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("JARVIS_BRAVE_SEARCH_API_KEY", "brave-secret-key")

    safe = BraveSearchProvider().status().model_dump(mode="json")

    assert safe["configured"] is True
    assert "brave-secret-key" not in str(safe)


def test_brave_search_parses_results_without_real_network(monkeypatch) -> None:
    monkeypatch.setattr(
        "jarvis.web_search.providers.urlopen",
        lambda request, timeout=8.0: _FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "Python Testing",
                            "url": "https://example.com/testing",
                            "description": "Pytest fixtures and examples.",
                            "age": "2026-01-01",
                        }
                    ]
                }
            }
        ),
    )
    provider = BraveSearchProvider(api_key="brave-secret-key", enabled=True, max_results=5, timeout_seconds=8)

    result = provider.search("pytest fixtures", max_results=5)

    assert result.status == "ok"
    assert result.hits[0].title == "Python Testing"
    assert result.hits[0].source == "example.com"
    assert "brave-secret-key" not in str(result.model_dump(mode="json"))


def test_brave_search_blocks_secret_query_without_network(monkeypatch) -> None:
    called = False

    def _urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network should not be called")

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", _urlopen)
    provider = BraveSearchProvider(api_key="brave-secret-key", enabled=True)

    result = provider.search("busca sk-secret-token-value")

    assert result.status == "blocked"
    assert called is False


def test_grounded_prompt_contains_sources_and_identity_policy() -> None:
    response = WebSearchResponse(
        status="ok",
        provider="brave",
        query="noticias hoy",
        hits=[WebSearchHit(title="Fuente", url="https://example.com", snippet="Resumen", source="example.com", rank=1)],
    )
    prompt = build_grounded_web_prompt("noticias hoy", response)

    assert "Responde como Jarvis" in prompt
    assert "Fuente" in prompt
    assert "Encontre 1 fuente." in prompt


def test_grounded_prompt_limits_deduplicates_and_truncates_sources() -> None:
    long_snippet = "a" * 900
    response = WebSearchResponse(
        status="ok",
        provider="brave",
        query="noticias hoy",
        hits=[
            WebSearchHit(title="Uno", url="https://one.example/a", snippet=long_snippet, source="one.example", rank=1),
            WebSearchHit(title="Uno duplicado", url="https://one.example/b", snippet="duplicado", source="one.example", rank=2),
            WebSearchHit(title="Dos", url="https://two.example", snippet="b" * 200, source="two.example", rank=3),
            WebSearchHit(title="Tres", url="https://three.example", snippet="c" * 200, source="three.example", rank=4),
            WebSearchHit(title="Cuatro", url="https://four.example", snippet="d" * 200, source="four.example", rank=5),
        ],
    )

    prompt = build_grounded_web_prompt("noticias hoy", response, max_sources=3, snippet_chars=120)

    assert "one.example" in prompt
    assert "two.example" in prompt
    assert "three.example" in prompt
    assert "four.example" not in prompt
    assert "Uno duplicado" not in prompt
    assert long_snippet not in prompt
    assert "Encontre 5 fuentes y use 3 para redactar la respuesta." in prompt


def test_source_count_message_uses_real_counts() -> None:
    assert source_count_message(0) == "No encontre fuentes confiables para esta busqueda."
    assert source_count_message(1) == "Encontre 1 fuente."
    assert source_count_message(2) == "Encontre 2 fuentes."
    assert source_count_message(5, 3) == "Encontre 5 fuentes y use 3 para redactar la respuesta."


def test_web_cli_status_and_search_without_key(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_BRAVE_SEARCH_API_KEY", "")
    monkeypatch.setenv("JARVIS_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setenv("JARVIS_WEB_SEARCH_PROVIDER", "brave")
    runner = CliRunner()

    status = runner.invoke(app, ["web", "status"])
    search = runner.invoke(app, ["web", "search", "noticias de tecnologia hoy"])

    assert status.exit_code == 0
    assert "API key: configured false" in status.stdout
    assert search.exit_code == 0
    assert "BRAVE_SEARCH_API_KEY is present but empty" in search.stdout


def test_canonical_brave_key_has_priority_and_is_sent_only_as_header(monkeypatch) -> None:
    captured = {}

    def _urlopen(request, timeout):
        captured["token"] = request.get_header("X-subscription-token")
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _FakeResponse({"web": {"results": []}})

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "canonical-key")
    monkeypatch.setenv("JARVIS_BRAVE_SEARCH_API_KEY", "legacy-key")
    monkeypatch.setenv("JARVIS_WEB_SEARCH_ENABLED", "true")
    monkeypatch.setattr("jarvis.web_search.providers.urlopen", _urlopen)

    provider = build_web_search_provider(Settings(_env_file=None))
    result = provider.search("café NVIDIA", max_results=1)

    assert result.status == "empty"
    assert captured["token"] == "canonical-key"
    assert "canonical-key" not in captured["url"]
    assert "caf%C3%A9+NVIDIA" in captured["url"]
    assert captured["timeout"] == 8.0


def test_brave_diagnoses_empty_missing_env_and_misnamed_key(monkeypatch, tmp_path) -> None:
    missing_env = tmp_path / ".env"
    monkeypatch.setattr("jarvis.web_search.providers._ENV_FILE", missing_env)
    for name in ("BRAVE_SEARCH_API_KEY", "JARVIS_BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    missing = BraveSearchProvider(enabled=True, settings=Settings(_env_file=None)).status()
    assert missing.configuration_state == "env_file_missing"

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    empty = BraveSearchProvider(enabled=True, settings=Settings(_env_file=None)).status()
    assert empty.configuration_state == "empty_key"

    monkeypatch.delenv("BRAVE_SEARCH_API_KEY")
    monkeypatch.setenv("BRAVE_SEARCH_KEY", "wrong-name")
    misnamed = BraveSearchProvider(enabled=True, settings=Settings(_env_file=None)).status()
    assert misnamed.configuration_state == "misnamed_variable"
    assert "BRAVE_SEARCH_KEY" in misnamed.message


def test_brave_is_primary_but_does_not_block_startup_without_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("jarvis.web_search.providers._ENV_FILE", tmp_path / ".env")
    for name in ("BRAVE_SEARCH_API_KEY", "JARVIS_BRAVE_SEARCH_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        _env_file=None,
        web_search_provider="brave",
        web_search_enabled=None,
        brave_search_api_key=None,
    )

    provider = build_web_search_provider(settings)
    status = provider.status()
    result = provider.search("OpenAI")

    assert status.provider == "brave"
    assert status.enabled is True
    assert status.configured is False
    assert result.status == "unavailable"
    assert result.error_code in {"missing_key", "env_file_missing"}


def test_brave_maps_http_failures_without_exposing_key(monkeypatch) -> None:
    cases = [
        (401, "unauthorized", "authentication_failed"),
        (403, "forbidden", "forbidden"),
        (429, "rate_limited", "rate_limited"),
        (503, "server_error", "server_error"),
    ]
    provider = BraveSearchProvider(api_key="private-key", enabled=True)
    for code, expected_status, error_code in cases:
        headers = {"Retry-After": "12"} if code == 429 else {}

        def _raise(*_args, _code=code, _headers=headers, **_kwargs):
            raise HTTPError("https://api.search.brave.com", _code, "failure", _headers, None)

        monkeypatch.setattr("jarvis.web_search.providers.urlopen", _raise)
        result = provider.search("OpenAI")
        assert result.status == expected_status
        assert result.error_code == error_code
        assert result.http_status == code
        assert "private-key" not in str(result.model_dump(mode="json"))
        if code == 429:
            assert result.retry_after_seconds == 12


def test_brave_distinguishes_timeout_network_and_invalid_response(monkeypatch) -> None:
    provider = BraveSearchProvider(api_key="private-key", enabled=True)

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(socket.timeout()))
    assert provider.search("OpenAI").status == "timeout"

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")))
    assert provider.search("OpenAI").status == "network_error"

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", lambda *_args, **_kwargs: _RawResponse(b"not-json"))
    assert provider.search("OpenAI").status == "invalid_response"

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("private detail")))
    unexpected = provider.search("OpenAI")
    assert unexpected.status == "internal_error"
    assert "private detail" not in unexpected.message


def test_brave_health_is_cached_and_reports_four_dimensions(monkeypatch) -> None:
    calls = 0

    def _urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _FakeResponse({"web": {"results": []}})

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", _urlopen)
    provider = BraveSearchProvider(api_key="private-key", enabled=True, health_ttl_seconds=300)

    first = provider.check_health()
    second = provider.check_health()

    assert calls == 1
    assert first.configured is True
    assert first.authenticated is True
    assert first.reachable is True
    assert first.working is True
    assert second.checked_at == first.checked_at


def test_timeout_health_does_not_disable_future_searches(monkeypatch) -> None:
    calls = 0

    def _urlopen(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise socket.timeout()
        return _FakeResponse({"web": {"results": []}})

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", _urlopen)
    provider = BraveSearchProvider(api_key="private-key", enabled=True)

    assert provider.search("first").status == "timeout"
    assert provider.search("second").status == "empty"
    assert calls == 2


def test_benchmark_cli_does_not_print_secrets(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JARVIS_BRAVE_SEARCH_API_KEY", "brave-secret-key")
    monkeypatch.setenv("JARVIS_WEB_SEARCH_ENABLED", "false")
    runner = CliRunner()

    result = runner.invoke(app, ["benchmark"])

    assert result.exit_code == 0
    assert "Jarvis Benchmark" in result.stdout
    assert "brave-secret-key" not in result.stdout
    assert "OpenAI: blocked" in result.stdout
    assert "Gemini: blocked" in result.stdout
