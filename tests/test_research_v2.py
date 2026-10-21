from __future__ import annotations

import json

from jarvis.config import Settings
from jarvis.file_runtime.runtime import FileNavigator
from jarvis.web_search import (
    CitationRecord,
    ClaimRequirementStatus,
    ClaimSupportStatus,
    EvidenceChunk,
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
    detect_freshness,
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


def test_immigration_report_extracts_bounded_requirements_and_freshness() -> None:
    question = (
        "investiga sobre la violacion a los derechos humanos y el maltrato que estan "
        "sufriendo los inmigrantes a estados unidos, al final dame un informe y las "
        "fuentes consultadas en formato APA"
    )
    intent = parse_research_intent(question, SearchMode.DEEP)
    descriptions = [item.description for item in intent.requirements]

    assert "documented human-rights violations and legal context" in descriptions
    assert "reported mistreatment and detention conditions" in descriptions
    assert len(descriptions) == 4
    assert "fuentes" not in intent.topic.casefold()
    assert detect_freshness(question).requires_fresh_information is True


def test_claim_validation_does_not_conflate_cancer_risk_with_persistence(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run(
        "/deepresearch busca causas y efectos del cancer y por qué a veces no se quita",
        mode="deep",
    )
    session.evidence = [
        EvidenceChunk(
            evidence_id="risk",
            source_id="risk-source",
            citation_id="S1",
            url="https://cancer.gov/risk",
            title="Cancer risk factors",
            domain="cancer.gov",
            text="Inherited mutations and environmental exposures can increase the risk of developing cancer.",
            requirement_ids=["R2"],
        ),
        EvidenceChunk(
            evidence_id="residual",
            source_id="residual-source",
            citation_id="S2",
            url="https://cancer.gov/recurrence",
            title="Cancer recurrence",
            domain="cancer.gov",
            text="Cancer can recur when residual malignant cells remain after treatment. Treatment resistance can also allow disease to persist.",
            requirement_ids=["R4"],
        ),
    ]
    orchestrator._update_requirement_coverage(session)  # noqa: SLF001
    persistence = next(
        item for item in session.intent.requirements if "persistence" in item.description
    )

    validation = orchestrator.validate_claim_support(
        "La predisposición genética hace que un cáncer existente persista después del tratamiento [S1].",
        session,
    )

    assert validation.valid is False
    assert "risk" not in persistence.evidence_ids
    assert "residual" in persistence.evidence_ids
    assert validation.claims[0].status is ClaimSupportStatus.UNSUPPORTED
    assert validation.claims[0].suggested_citation_ids == ["S2"]
    assert set(orchestrator.uncovered_claim_requirements(validation, session)) == {
        "causes and risk factors",
        "effects and consequences",
        persistence.description,
    }


def test_supported_claim_can_still_be_off_requirement(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run(
        "/deepresearch busca causas y efectos del cancer y por qué a veces no se quita",
        mode="deep",
    )
    session.evidence = [
        EvidenceChunk(
            evidence_id="screening",
            source_id="official",
            citation_id="S1",
            url="https://example.gov/screening",
            title="Screening harms",
            domain="example.gov",
            text="El sobrediagnóstico de cáncer puede producir daño psicosocial tras el screening.",
        )
    ]

    validation = orchestrator.validate_claim_support(
        "## Efectos y consecuencias\n"
        "El sobrediagnóstico de cáncer puede producir daño psicosocial tras el screening [S1].",
        session,
    )

    assert validation.claims[0].status is ClaimSupportStatus.SUPPORTED
    assert validation.claims[0].requirement_status is ClaimRequirementStatus.OFF_REQUIREMENT
    assert validation.valid is False


def test_requirement_relevance_distinguishes_context_and_direct_claims(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run(
        "/deepresearch busca causas y efectos del cancer y por qué a veces no se quita",
        mode="deep",
    )
    session.evidence = [
        EvidenceChunk(
            evidence_id="effect",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/effects",
            title="Cancer effects",
            domain="cancer.gov",
            text="El cáncer puede invadir órganos cercanos, causar dolor y dañar su función.",
        ),
        EvidenceChunk(
            evidence_id="toxicity",
            source_id="official",
            citation_id="S2",
            url="https://cancer.gov/treatment",
            title="Treatment effects",
            domain="cancer.gov",
            text="Los efectos secundarios del tratamiento pueden incluir fatiga y náuseas.",
        ),
    ]

    validation = orchestrator.validate_claim_support(
        "## Efectos y consecuencias\n"
        "El cáncer puede invadir órganos cercanos, causar dolor y dañar su función [S1]. "
        "Los efectos secundarios del tratamiento pueden incluir fatiga y náuseas [S2].",
        session,
    )

    assert validation.claims[0].requirement_status is ClaimRequirementStatus.DIRECTLY_RELEVANT
    assert validation.claims[1].requirement_status is ClaimRequirementStatus.SUPPORTING_CONTEXT
    assert validation.directly_relevant == 1
    assert validation.supporting_context == 1


def test_stronger_equivalent_source_replaces_weak_citation(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga efectos del cancer", mode="deep")
    common = "El cáncer puede invadir órganos cercanos y causar dolor."
    session.evidence = [
        EvidenceChunk(
            evidence_id="news",
            source_id="bbc",
            citation_id="S1",
            url="https://bbc.com/example",
            title="Explainer",
            domain="bbc.com",
            text=common,
            relevance_score=0.8,
        ),
        EvidenceChunk(
            evidence_id="nci",
            source_id="nci",
            citation_id="S2",
            url="https://cancer.gov/example",
            title="Cancer effects",
            domain="cancer.gov",
            text=common,
            relevance_score=0.8,
            evidence_type="official",
        ),
    ]
    answer = "## Efectos y consecuencias\nEl cáncer puede invadir órganos cercanos y causar dolor [S1]."

    validation = orchestrator.validate_claim_support(answer, session)
    repaired, changes = orchestrator.replace_weak_claim_citations(answer, validation)

    assert validation.claims[0].preferred_citation_ids == ["S2"]
    assert validation.claims[0].replaced_citation_ids == ["S1"]
    assert "[S2]" in repaired and "[S1]" not in repaired
    assert changes[0]["claim_id"] == "C1"


def test_claim_validation_supports_residual_disease_and_detects_overstatement(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga recurrencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="residual",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/recurrence",
            title="Recurrence",
            domain="cancer.gov",
            text="Residual malignant cells can remain after treatment and can lead to cancer recurrence.",
        ),
        EvidenceChunk(
            evidence_id="hedged",
            source_id="study",
            citation_id="S2",
            url="https://example.edu/study",
            title="Study",
            domain="example.edu",
            text="Tumor heterogeneity may contribute to treatment resistance and recurrence.",
        ),
    ]

    validation = orchestrator.validate_claim_support(
        "Las células malignas residuales pueden causar recurrencia [S1]. "
        "La heterogeneidad tumoral causa resistencia al tratamiento y recurrencia [S2].",
        session,
    )

    assert validation.claims[0].status is ClaimSupportStatus.SUPPORTED
    assert validation.claims[1].status is ClaimSupportStatus.PARTIALLY_SUPPORTED
    assert validation.valid is False


def test_claim_validation_aggregates_cited_evidence_chunks(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga por que persiste el cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="residual",
            source_id="review",
            citation_id="S1",
            url="https://pmc.ncbi.nlm.nih.gov/review",
            title="Residual disease",
            domain="pmc.ncbi.nlm.nih.gov",
            text="Residual malignant cells can remain after cancer treatment and cause persistent disease.",
        ),
        EvidenceChunk(
            evidence_id="resistance",
            source_id="review",
            citation_id="S1",
            url="https://pmc.ncbi.nlm.nih.gov/review",
            title="Treatment resistance",
            domain="pmc.ncbi.nlm.nih.gov",
            text="Drug resistance allows malignant cells to survive therapy and can lead to recurrence.",
        ),
    ]

    validation = orchestrator.validate_claim_support(
        "Las células malignas residuales y la resistencia al tratamiento pueden causar "
        "persistencia y recurrencia del cáncer [S1].",
        session,
    )

    assert validation.valid is True
    assert validation.claims[0].status is ClaimSupportStatus.SUPPORTED
    assert set(validation.claims[0].evidence_ids) == {"residual", "resistance"}
    assert orchestrator.uncovered_claim_requirements(validation, session) == []


def test_claim_validation_rejects_explicitly_admitted_extrapolation(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga causas del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="definition",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/definition",
            title="Cancer definition",
            domain="cancer.gov",
            text="Cancer is a disease in which abnormal cells grow without control.",
        )
    ]

    validation = orchestrator.validate_claim_support(
        "Aunque la evidencia no lo declara explícitamente, esta definición implica "
        "que una exposición ambiental causa cáncer [S1].",
        session,
    )

    assert validation.valid is False
    assert validation.claims[0].status is ClaimSupportStatus.UNSUPPORTED


def test_evolutionary_susceptibility_does_not_support_treated_cancer_persistence(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch por que el cancer a veces no se quita", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="evolution",
            source_id="news",
            citation_id="S1",
            url="https://example.org/evolution",
            title="Cancer and evolution",
            domain="example.org",
            text=(
                "Natural selection may not remove mutations whose harmful effects appear "
                "after the reproductive period, so cancer susceptibility persists."
            ),
        )
    ]

    validation = orchestrator.validate_claim_support(
        "La persistencia o recurrencia del cáncer se explica por la selección natural y "
        "por efectos que aparecen después del período reproductivo [S1].",
        session,
    )

    assert validation.valid is False
    assert validation.claims[0].status is ClaimSupportStatus.UNSUPPORTED


def test_missing_persistence_requirement_can_be_repaired_from_direct_nci_evidence(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch por que el cancer a veces no se quita", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="nci-recurrence",
            source_id="nci",
            citation_id="S7",
            url="https://cancer.gov/types/recurrent-cancer",
            title="Recurrent Cancer",
            domain="cancer.gov",
            text=(
                "Recurrent cancer starts with cancer cells that the first treatment did not "
                "fully remove or destroy. A small number of cancer cells survived treatment "
                "and over time grew into tumors that can be detected."
            ),
        )
    ]

    repaired, repairs = orchestrator.repair_missing_claim_requirements(
        "# Informe sobre Persistencia del Cáncer\n\n"
        "## Contexto\nTexto introductorio.\n\n"
        "## Persistencia y Razones de la No Resolución",
        session,
        ["persistence and why it may not resolve"],
    )
    validation = orchestrator.validate_claim_support(repaired, session)

    assert repairs[0]["citation_id"] == "S7"
    assert "tratamiento inicial" in repaired
    assert repaired.index("tratamiento inicial") > repaired.index(
        "## Persistencia y Razones de la No Resolución"
    )
    assert validation.valid is True
    persistence_requirement = next(
        item
        for item in session.intent.requirements
        if "persistence" in item.description
    )
    assert validation.claims[0].requirement_id == persistence_requirement.requirement_id


def test_claim_extraction_keeps_text_below_markdown_heading(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga recurrencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="recurrence",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/recurrence",
            title="Recurrence",
            domain="cancer.gov",
            text="Residual malignant cells can remain after treatment and cause cancer recurrence.",
        )
    ]

    validation = orchestrator.validate_claim_support(
        "## Persistencia\nLas células malignas residuales pueden causar recurrencia [S1].",
        session,
    )

    assert len(validation.claims) == 1
    assert validation.claims[0].status is ClaimSupportStatus.SUPPORTED


def test_claim_validation_flags_direct_contradiction(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga recurrencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="negative",
            source_id="study",
            citation_id="S1",
            url="https://example.edu/study",
            title="Study",
            domain="example.edu",
            text="This inherited variant does not cause cancer recurrence after treatment.",
        )
    ]

    validation = orchestrator.validate_claim_support(
        "Esta variante hereditaria causa recurrencia del cáncer después del tratamiento [S1].",
        session,
    )

    assert validation.claims[0].status is ClaimSupportStatus.CONTRADICTED


def test_claim_validation_keeps_facts_below_markdown_headings(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga recurrencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="residual",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/recurrence",
            title="Recurrence",
            domain="cancer.gov",
            text="Residual malignant cells can remain after treatment and can lead to cancer recurrence.",
        )
    ]

    validation = orchestrator.validate_claim_support(
        "## Persistencia y razones de la no resolución\n"
        "Las células malignas residuales pueden causar recurrencia [S1].",
        session,
    )

    assert len(validation.claims) == 1
    assert validation.claims[0].status is ClaimSupportStatus.SUPPORTED


def test_final_claim_prune_removes_only_unverified_sentences(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga persistencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="residual",
            source_id="official",
            citation_id="S1",
            url="https://cancer.gov/recurrence",
            title="Recurrence",
            domain="cancer.gov",
            text="Residual malignant cells can remain after treatment and cause cancer recurrence.",
        ),
        EvidenceChunk(
            evidence_id="risk",
            source_id="risk",
            citation_id="S2",
            url="https://cancer.gov/risk",
            title="Risk",
            domain="cancer.gov",
            text="Inherited mutations can increase the risk of developing cancer.",
        ),
    ]
    answer = (
        "## Persistencia\n"
        "Las células malignas residuales pueden causar recurrencia [S1]. "
        "La predisposición hereditaria hace persistir un cáncer tratado [S2]."
    )
    validation = orchestrator.validate_claim_support(answer, session)

    repaired, removed = orchestrator.remove_unverified_claims(answer, validation)

    assert removed == ["C2"]
    assert "residuales" in repaired
    assert "predisposición" not in repaired
    assert orchestrator.validate_claim_support(repaired, session).valid is True


def test_final_claim_prune_matches_claim_that_includes_inline_subheading(tmp_path) -> None:
    orchestrator = WebResearchOrchestrator(_Provider(), _settings(tmp_path))
    session = orchestrator.run("/deepresearch investiga persistencia del cancer", mode="deep")
    session.evidence = [
        EvidenceChunk(
            evidence_id="risk",
            source_id="risk",
            citation_id="S1",
            url="https://cancer.gov/risk",
            title="Risk",
            domain="cancer.gov",
            text="Inherited mutations can increase the risk of developing cancer.",
        )
    ]
    answer = (
        "## Persistencia\n"
        "**Mecanismo de persistencia**\n"
        "La predisposición hereditaria hace persistir un cáncer tratado [S1]."
    )
    validation = orchestrator.validate_claim_support(answer, session)

    repaired, removed = orchestrator.remove_unverified_claims(answer, validation)

    assert validation.unsupported == 1
    assert removed == ["C1"]
    assert "predisposición" not in repaired


def test_unsupported_required_claim_triggers_one_targeted_gap_search(tmp_path) -> None:
    provider = _Provider()
    orchestrator = WebResearchOrchestrator(provider, _settings(tmp_path))
    session = orchestrator.run(
        "/deepresearch busca causas y efectos del cancer y por qué a veces no se quita",
        mode="deep",
    )
    session.evidence = [
        EvidenceChunk(
            evidence_id="risk",
            source_id="risk-source",
            citation_id="S1",
            url="https://example.org/risk",
            title="Risk factors",
            domain="example.org",
            text="Environmental exposure can increase the risk of developing cancer.",
            requirement_ids=["R2"],
        )
    ]
    validation = orchestrator.validate_claim_support(
        "La resistencia al tratamiento puede permitir que el cáncer persista [S1].",
        session,
    )
    calls_before = len(provider.calls)

    result = orchestrator.search_claim_evidence_gaps(session, validation)

    assert len(provider.calls) == calls_before + 1
    assert result["queries"] == [provider.calls[-1]]
    assert "treatment resistance" in provider.calls[-1]
    assert session.metrics.queries_count == len(session.queries)


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
        if "ydc-index.io" in request.full_url:
            return _Response({"results": {"web": [{"title": "You result", "url": "https://you.example/a", "description": "answer"}]}})
        return _Response({"results": [{"title": "Exa result", "url": "https://exa.example/a", "highlights": ["technical evidence"]}]})

    monkeypatch.setattr("jarvis.web_search.providers.urlopen", open_fake)
    you = YouSearchProvider(api_key="you-secret", settings=Settings(_env_file=None))
    exa = ExaSearchProvider(api_key="exa-secret", settings=Settings(_env_file=None))

    assert you.search("public query", max_results=2).status == "ok"
    assert exa.search("technical query", max_results=2).hits[0].highlights == ["technical evidence"]
    assert captured[0][2]["count"] == 2
    assert captured[0][0] == "https://ydc-index.io/v1/search"
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


def test_bbc_social_author_is_rejected_and_publisher_is_used(tmp_path) -> None:
    body = """<html><head><title>BBC report</title>
        <meta property="article:author" content="https://www.facebook.com/bbcnews.">
        <meta property="og:site_name" content="BBC News Mundo">
        <meta property="article:published_time" content="2021-04-01">
        </head><article>Verified evidence body.</article></html>""".encode()
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
        source_id="bbc",
        title="¿De dónde viene el cáncer?",
        url="https://bbc.com/mundo/noticias-56580528",
        normalized_url="https://bbc.com/mundo/noticias-56580528",
        domain="bbc.com",
    )

    source = fetcher.fetch(candidate)

    assert source.author is None
    assert source.organization == "BBC News Mundo"
    assert source.metadata_audit["rejected"][0]["field"] == "author"
    assert source.metadata_audit["publisher_fallbacks"]

    session = WebResearchOrchestrator(_Provider(), _settings(tmp_path)).run(
        "/deepresearch investiga cáncer y fuentes APA", mode="deep"
    )
    session.citations = [
        CitationRecord(
            citation_id="S1",
            source_id="bbc",
            title=candidate.title,
            url=candidate.url,
            domain=candidate.domain,
            published_at=source.published_at,
            author="https://www.facebook.com/bbcnews.",
            organization=source.organization,
        )
    ]
    session.intent.requested_format = "report with apa references"
    references = WebResearchOrchestrator(_Provider(), _settings(tmp_path)).format_references(
        session, ["S1"]
    )

    assert "facebook.com" not in references
    assert "BBC News Mundo. (2021)." in references


def test_conflicting_author_metadata_prefers_higher_confidence_not_first() -> None:
    body = b"""<html><head><title>Report</title>
        <meta property="article:author" content="Organization Feed">
        <meta name="author" content="Ada Example">
        <meta property="og:site_name" content="Example Institute">
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
    source = fetcher.fetch(
        SourceCandidate(
            source_id="source",
            title="Report",
            url="https://example.com/report",
            normalized_url="https://example.com/report",
            domain="example.com",
        )
    )

    assert source.author == "Ada Example"
    assert source.metadata_provenance["author"]["source"] == "meta:author"
    assert source.metadata_audit["conflicts"][0]["field"] == "author"


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
