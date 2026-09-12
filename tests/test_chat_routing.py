from __future__ import annotations

import pytest

from jarvis.cognition.capabilities import install_cognitive_capabilities
from jarvis.cognition.orchestrator import CognitiveOrchestrator
from jarvis.config import Settings
from jarvis.core.capabilities import CapabilityRegistry
from jarvis.desktop import build_desktop_runtime
from jarvis.modules.research_module import ResearchModule
from jarvis.modules.science_module import ScienceModule
from jarvis.modules.security_module import SecurityModule
from jarvis.modules.system_module import SystemModule
from jarvis.modules.vision_module import VisionModule
from jarvis.modules.voice_interface_module import VoiceInterfaceModule
from jarvis.models_runtime.base import ModelResponse, ProviderKind
from jarvis.web_search import RawPage, SourceFetcher, WebSearchHit, WebSearchResponse


class _NoOpDependency:
    def __getattr__(self, _name):
        raise AssertionError("unexpected dependency call during routing test")


class _ModelStub:
    def infer(self, _request):
        raise AssertionError("model classification should not run for deterministic routing tests")


def _build_orchestrator() -> CognitiveOrchestrator:
    capabilities = CapabilityRegistry()
    install_cognitive_capabilities(capabilities)
    ResearchModule().register_capabilities(capabilities)
    ScienceModule(_NoOpDependency()).register_capabilities(capabilities)
    SecurityModule(_NoOpDependency()).register_capabilities(capabilities)
    SystemModule(_NoOpDependency()).register_capabilities(capabilities)
    VisionModule(_NoOpDependency()).register_capabilities(capabilities)
    VoiceInterfaceModule(_NoOpDependency()).register_capabilities(capabilities)
    return CognitiveOrchestrator(
        router=_NoOpDependency(),
        memory=_NoOpDependency(),
        capabilities=capabilities,
        models=_ModelStub(),
    )


def test_orchestrator_keeps_general_prompt_in_general_chat() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("que es la entropia", {}, correlation_id="routing-general") == "general_chat"  # noqa: SLF001


def test_orchestrator_does_not_route_grounding_prompt_to_ui_awareness_without_visual_context() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("encuentra texto guardar", {}, correlation_id="routing-ui") == "general_chat"  # noqa: SLF001


def test_orchestrator_does_not_route_general_search_language_to_research() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("busca el significado de entropia", {}, correlation_id="routing-search") == "general_chat"  # noqa: SLF001


def test_orchestrator_routes_explicit_visual_request_to_ui_awareness() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("encuentra el texto guardar en la pantalla", {}, correlation_id="routing-screen") == "ui_awareness"  # noqa: SLF001


def test_orchestrator_routes_explicit_desktop_vision_prompts_to_screen_read() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("puedes ver lo que hay en mi escritorio", {}, correlation_id="routing-desktop") == "screen_read"  # noqa: SLF001
    assert orchestrator._infer_intent("que hay en mi pantalla", {}, correlation_id="routing-screen-view") == "screen_read"  # noqa: SLF001
    assert orchestrator._infer_intent("que ves en la ventana actual", {}, correlation_id="routing-window-view") == "screen_read"  # noqa: SLF001


def test_orchestrator_keeps_metaphorical_vision_language_in_general_chat() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("ves lo que digo", {}, correlation_id="routing-metaphor") == "general_chat"  # noqa: SLF001


def test_orchestrator_routes_science_and_security_to_specialized_runtimes() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator._infer_intent("calcula derivada de x^2", {}, correlation_id="routing-science") == "science"  # noqa: SLF001
    assert orchestrator._infer_intent("evalua esta contrasena password123", {}, correlation_id="routing-security") == "security"  # noqa: SLF001


@pytest.mark.parametrize("provider_kind", [ProviderKind.LOCAL, ProviderKind.REMOTE])
def test_chat_engine_routes_research_request_through_shared_pipeline(tmp_path, provider_kind) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
        local_runtime_autostart=False,
        nvidia_api_key=None,
        web_search_enabled=False,
    )
    app, desktop = build_desktop_runtime(settings)
    requests = []

    def infer(request):
        requests.append(request)
        return ModelResponse(
            provider_name="nemotron-test",
            provider_kind=provider_kind,
            logical_model=request.logical_model or "general_assistant",
            model_name="nemotron-test",
            content="Investigación completada con evidencia. [S1]",
            latency_ms=1.0,
        )

    class FakeWeb:
        def search(self, query, *, max_results=5):  # noqa: ARG002
            return WebSearchResponse(
                status="ok",
                provider="fake",
                query=query,
                hits=[
                    WebSearchHit(
                        title="Agujeros negros",
                        url="https://example.com/black-holes",
                        snippet="Evidencia astronómica",
                        rank=1,
                    )
                ],
            )

    app.web_research_orchestrator.provider = FakeWeb()
    app.web_research_orchestrator.fetcher = SourceFetcher(
        resolver=lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
        transport=lambda url, _timeout, _max_bytes: RawPage(
            url=url,
            content_type="text/html",
            body=b"<article>Agujeros negros evidencia astronomica fuente primaria</article>",
            headers={"content-type": "text/html"},
        ),
    )
    app.model_service.infer = infer  # type: ignore[method-assign]
    try:
        response = desktop.send_chat("investiga agujeros negros")
        assert response.message.content
        assert "orchestrated" not in response.message.content
        assert response.raw_result["research"]["session_id"]
        assert len(requests) == 1
        assert requests[0].metadata["search_mode"] == "normal"
    finally:
        app.stop()


def test_chat_engine_returns_writing_guard_feedback(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ui_backend_kind="in_memory",
    )
    app, desktop = build_desktop_runtime(settings)
    try:
        response = desktop.send_chat("continua mi texto actual")
        assert "contexto suficiente" in response.message.content.casefold()
        assert response.raw_result.get("ok") is False
    finally:
        app.stop()
