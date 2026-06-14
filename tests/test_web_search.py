from __future__ import annotations

import json

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.web_search import BraveSearchProvider, WebSearchHit, WebSearchResponse, build_grounded_web_prompt, sanitize_web_query, should_use_web_search


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
    assert "Brave Search API key not configured" in search.stdout


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
