from __future__ import annotations

import json

from jarvis.config import Settings
from jarvis.file_runtime.runtime import FileNavigator
from jarvis.web_search import (
    ExaSearchProvider,
    ResearchDepth,
    SearchMode,
    RawPage,
    SearXNGSearchProvider,
    SourceCandidate,
    SourceFetcher,
    SourceScope,
    WebProviderRouter,
    WebResearchOrchestrator,
    WebSearchHit,
    WebSearchProviderStatus,
    WebSearchResponse,
    YouSearchProvider,
    parse_research_intent,
    public_web_topic,
    resolve_auto_scope,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _Provider:
    def __init__(self, name: str = "fake", *, status: str = "ok") -> None:
        self.provider = name
        self.role = "test"
        self.result_status = status
        self.calls: list[str] = []

    def status(self) -> WebSearchProviderStatus:
        return WebSearchProviderStatus(provider=self.provider, enabled=True, configured=True, available=True)

    def check_health(self, *, force: bool = False) -> WebSearchProviderStatus:
        del force
        return self.status()

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        self.calls.append(query)
        if self.result_status != "ok":
            return WebSearchResponse(status=self.result_status, provider=self.provider, query=query)
        slug = len(self.calls)
        return WebSearchResponse(
            status="ok",
            provider=self.provider,
            query=query,
            hits=[WebSearchHit(
                title=f"Cancer research evidence {slug}",
                url=f"https://evidence{slug}.example/report",
                snippet="Cancer definition causes risk effects consequences chronic persistence evidence conclusion.",
                provider=self.provider,
                rank=1,
                content="Cancer definition and context. Causes and risk factors vary. Effects and consequences can persist and become chronic. Evidence supports this conclusion.",
            )],
        )


class _Files:
    def __init__(self, text: str = "Brave Search appears in the Jarvis architecture notes.") -> None:
        self.text = text
        self.calls: list[tuple[str, str | None]] = []

    def search_content(self, query: str, *, path=None, **_kwargs):
        self.calls.append((query, path))
        return {"items": [{"path": "C:/workspace/notes.md", "name": "notes.md", "excerpt": self.text, "score": 1.0}]}


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        embeddings_enabled=False,
        **overrides,
    )


def test_depth_and_source_scope_are_independent() -> None:
    files = parse_research_intent("/deepresearch busca en mis archivos todo sobre Jarvis", SearchMode.DEEP)
    web = parse_research_intent("/deepresearch investiga modelos multimodales actuales", SearchMode.DEEP)
    combined = parse_research_intent("/deepresearch compara mis notas con informacion actual de NVIDIA", SearchMode.DEEP)

    assert (files.depth, files.source_scope) == (ResearchDepth.DEEP, SourceScope.FILES)
    assert web.depth is ResearchDepth.DEEP
    assert resolve_auto_scope(web, web_needed=True) is SourceScope.WEB
    assert (combined.depth, combined.source_scope) == (ResearchDepth.DEEP, SourceScope.FILES_AND_WEB)
    assert "mis notas" not in public_web_topic(combined).casefold()


def test_files_only_and_deep_files_never_call_web(tmp_path) -> None:
    provider = _Provider()
    files = _Files("Jarvis context and definition with an evidence-based conclusion about Brave Search.")
    session = WebResearchOrchestrator(provider, _settings(tmp_path), file_navigator=files).run(
        "/deepresearch busca en mis archivos dónde mencioné Brave Search", mode="deep"
    )

    assert session.source_scope is SourceScope.FILES
    assert session.depth is ResearchDepth.DEEP
    assert session.metrics.file_searches == 1
    assert session.metrics.provider_requests == {}
    assert provider.calls == []
    assert session.evidence[0].source_type == "file"


def test_file_miss_does_not_silently_fallback_to_web(tmp_path) -> None:
    provider = _Provider()
    files = _Files()
    files.search_content = lambda *_args, **_kwargs: {"items": []}

    session = WebResearchOrchestrator(provider, _settings(tmp_path), file_navigator=files).run(
        "busca en mis archivos dónde mencioné un unicornio", mode="normal"
    )

    assert session.stop_reason == "no_file_evidence"
    assert session.status == "insufficient_evidence"
    assert provider.calls == []


def test_files_and_web_combines_provenance_without_leaking_file_text(tmp_path) -> None:
    private_marker = "PRIVATE_INTERNAL_MARKER_7391"
    provider = _Provider()
    files = _Files(f"RTX 5070 notes {private_marker}; comparison context and conclusion.")
    session = WebResearchOrchestrator(provider, _settings(tmp_path), file_navigator=files).run(
        "/deepresearch compara mis notas sobre RTX 5070 con informacion actual de NVIDIA", mode="deep"
    )

    assert session.source_scope is SourceScope.FILES_AND_WEB
    assert {item.source_type for item in session.evidence} == {"file", "web"}
    assert provider.calls
    assert all(private_marker not in query for query in provider.calls)
    assert all("mis notas" not in query.casefold() for query in provider.calls)


def test_requirement_extraction_and_query_pollution_isolation() -> None:
    intent = parse_research_intent(
        "/deepresearch busca causas y efectos del cancer y por qué a veces no se quita, dame un informe y fuentes en formato APA",
        SearchMode.DEEP,
    )
    descriptions = {item.description for item in intent.requirements}

    assert {"causes and risk factors", "effects and consequences", "persistence and why it may not resolve"} <= descriptions
    assert "apa" not in intent.topic.casefold()
    assert "referencias" not in intent.topic.casefold()


def test_router_falls_back_and_deep_uses_multiple_providers() -> None:
    unavailable = _Provider("you", status="server_error")
    exa = _Provider("exa")
    brave = _Provider("brave")
    router = WebProviderRouter([unavailable, exa, brave])

    response = router.search_for("technical model benchmark", max_results=3, depth=ResearchDepth.DEEP)

    assert response.status == "ok"
    assert response.metadata["providers_used"] == ["exa", "you", "brave"]
    assert {hit.provider for hit in response.hits} == {"exa", "brave"}


def test_you_and_exa_use_official_post_shapes(monkeypatch) -> None:
    captured: list[tuple[str, str | None, dict]] = []

    def open_fake(request, timeout):
        del timeout
        captured.append((request.full_url, request.get_header("X-api-key"), json.loads(request.data)))
        if "you.com" in request.full_url:
            return _Response({"results": {"web": [{"title": "You result", "url": "https://you.example/a", "description": "answer"}]}})
        return _Response({"results": [{"title": "Exa result", "url": "https://exa.example/a", "highlights": ["technical evidence"]}]})

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", open_fake)
    you = YouSearchProvider(api_key="you-secret", settings=Settings(_env_file=None))
    exa = ExaSearchProvider(api_key="exa-secret", settings=Settings(_env_file=None))

    assert you.search("public query", max_results=2).status == "ok"
    assert exa.search("technical query", max_results=2).hits[0].highlights == ["technical evidence"]
    assert captured[0][2]["count"] == 2
    assert captured[1][2]["numResults"] == 2
    assert "highlights" in captured[1][2]["contents"]
    assert all("secret" not in url for url, _header, _payload in captured)


def test_searxng_is_optional_and_uses_json_api(monkeypatch) -> None:
    captured = {}

    def open_fake(request, timeout):
        captured.update(url=request.full_url, timeout=timeout)
        return _Response({"results": [{"title": "Local result", "url": "https://example.com/a", "content": "text"}]})

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", open_fake)
    offline = SearXNGSearchProvider(base_url="", settings=Settings(_env_file=None))
    configured = SearXNGSearchProvider(base_url="http://127.0.0.1:8080", settings=Settings(_env_file=None))

    assert offline.search("query").status == "unavailable"
    assert configured.search("query").status == "ok"
    assert "format=json" in captured["url"]


def test_reader_preserves_verified_html_metadata() -> None:
    body = b"""<html><head><title>Verified report</title>
        <meta name="author" content="Ada Example">
        <meta property="og:site_name" content="Example Institute">
        <meta property="article:published_time" content="2026-09-01">
        </head><article>Verified evidence body.</article></html>"""
    fetcher = SourceFetcher(
        resolver=lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
        transport=lambda url, _timeout, _max_bytes: RawPage(
            url=url,
            content_type="text/html",
            body=body,
            headers={"content-type": "text/html"},
        ),
    )
    candidate = SourceCandidate(
        source_id="source-test",
        title="Search title",
        url="https://example.com/report",
        normalized_url="https://example.com/report",
        domain="example.com",
        provider="you",
    )

    source = fetcher.fetch(candidate)

    assert source.author == "Ada Example"
    assert source.organization == "Example Institute"
    assert source.published_at == "2026-09-01"
    assert source.provider == "you"


def test_semantic_file_search_reads_content_not_only_filenames(tmp_path) -> None:
    notes = tmp_path / "meeting-notes.md"
    notes.write_text("The provider strategy uses Brave Search for public discovery.", encoding="utf-8")
    navigator = FileNavigator(tmp_path)

    result = navigator.search_content("donde mencione el buscador Brave", max_results=5)

    assert result["items"]
    assert result["items"][0]["name"] == "meeting-notes.md"
    assert "Brave Search" in result["items"][0]["excerpt"]


def test_research_stops_on_diminishing_information_gain(tmp_path) -> None:
    class Repeating(_Provider):
        def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
            self.calls.append(query)
            return WebSearchResponse(
                status="ok",
                provider=self.provider,
                query=query,
                hits=[WebSearchHit(
                    title="One narrow source",
                    url="https://same.example/report",
                    snippet="NVIDIA performance evidence",
                    provider=self.provider,
                    rank=1,
                    content="NVIDIA performance evidence only.",
                )],
            )

    provider = Repeating()
    session = WebResearchOrchestrator(
        provider,
        _settings(tmp_path, search_max_research_rounds=3, search_deep_max_queries=8),
    ).run("/deepresearch investiga NVIDIA GPU", mode="deep")

    assert session.metrics.search_rounds == 2
    assert session.metrics.information_gain[-1] == 0
    assert session.stop_reason == "diminishing_returns"
    assert session.status == "partial"
