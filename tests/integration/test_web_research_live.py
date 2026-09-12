from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.web_search import SearchMode, WebResearchOrchestrator, build_web_search_provider, build_web_search_router


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_WEB_RESEARCH_LIVE_TESTS") != "1",
    reason="set RUN_WEB_RESEARCH_LIVE_TESTS=1 to spend real Brave quota",
)
def test_real_brave_quick_search_reads_a_public_source(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "runtime")
    provider = build_web_search_provider(settings)
    if not provider.status().configured:
        pytest.skip("Brave Search is not configured")

    session = WebResearchOrchestrator(provider, settings).run(
        "Who is the current CEO of NVIDIA?",
        mode=SearchMode.QUICK,
    )

    assert session.metrics.raw_results > 0
    assert session.metrics.unique_domains > 0
    assert session.metrics.opened_sources > 0
    assert session.evidence
    assert session.citations


@pytest.mark.skipif(
    os.getenv("RUN_WEB_RESEARCH_LIVE_TESTS") != "1",
    reason="set RUN_WEB_RESEARCH_LIVE_TESTS=1 to spend real provider quota",
)
def test_configured_web_provider_router_health_and_search() -> None:
    router = build_web_search_router(Settings())
    configured = [status for status in router.statuses(force=True) if status.configured]
    if not configured:
        pytest.skip("No Web provider is configured")

    assert all(status.working is True for status in configured)
    response = router.search("OpenAI", max_results=1)
    assert response.status == "ok"
    assert response.hits


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("RUN_WEB_RESEARCH_E2E") != "1",
    reason="set RUN_WEB_RESEARCH_E2E=1 to use real Brave and model providers",
)
def test_real_deepresearch_chat_returns_system_controlled_citations(tmp_path) -> None:
    selected_llm_mode = os.getenv("WEB_RESEARCH_E2E_LLM_MODE", "nvidia_api")
    repository_root = Path(__file__).resolve().parents[2]
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=repository_root,
        research_allowed_roots=(repository_root,),
        llm_mode=selected_llm_mode,
        local_runtime_autostart=selected_llm_mode == "local",
        search_deep_max_queries=4,
        search_deep_max_pages=4,
        search_max_research_rounds=2,
    )
    app = build_application(settings)
    app.model_service.set_llm_mode(selected_llm_mode)
    if not app.web_research_orchestrator.provider.status().configured:
        pytest.skip("No Web provider is configured")
    app.start()
    desktop = DesktopRuntimeService(app)
    try:
        response = desktop.send_chat(
            "/deepresearch Compare the newest small open-source multimodal models.",
            metadata={"search_mode": "deep"},
        )
    finally:
        desktop.shutdown()
        app.stop()

    research = response.raw_result["research"]
    assert response.raw_result["search_mode"] == "deep"
    assert research["metrics"]["queries_count"] >= 2
    assert research["metrics"]["opened_sources"] >= 2
    assert research["sources"]
    assert response.raw_result["status"] in {"completed", "completed_with_repairs"}
    assert response.raw_result["citation_validation"]["invalid_ids"] == []
    assert response.raw_result["citation_validation"]["invalid_urls"] == []
    assert "[S" in response.message.content


@pytest.mark.e2e
@pytest.mark.skipif(
    os.getenv("RUN_CANCER_DEEP_RESEARCH_E2E") != "1",
    reason="set RUN_CANCER_DEEP_RESEARCH_E2E=1 for the real cancer regression",
)
def test_real_cancer_deepresearch_returns_a_synthesis_not_a_source_dump(
    tmp_path,
) -> None:
    selected_llm_mode = os.getenv("WEB_RESEARCH_E2E_LLM_MODE", "local")
    repository_root = Path(__file__).resolve().parents[2]
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=repository_root,
        research_allowed_roots=(repository_root,),
        llm_mode=selected_llm_mode,
        local_runtime_autostart=selected_llm_mode == "local",
    )
    app = build_application(settings)
    app.model_service.set_llm_mode(selected_llm_mode)
    if not app.web_research_orchestrator.provider.status().configured:
        pytest.skip("No Web provider is configured")
    app.start()
    desktop = DesktopRuntimeService(app)
    try:
        response = desktop.send_chat(
            "/deepresearch ¿qué es el cáncer y cómo se desarrolla?",
            metadata={"search_mode": "deep"},
        )
    finally:
        desktop.shutdown()
        app.stop()

    research = response.raw_result["research"]
    assert research["metrics"]["queries_count"] >= 2
    assert research["metrics"]["opened_sources"] >= 3
    assert response.raw_result["status"] in {"completed", "completed_with_repairs"}
    assert response.raw_result["citation_validation"]["valid"] is True
    assert response.raw_result["citation_validation"]["invalid_ids"] == []
    assert "[S" in response.message.content
    assert "Conservé únicamente las fuentes" not in response.message.content
