from pathlib import Path

from jarvis.research_runtime.models import (
    ResearchRunRequest,
    ResearchStepKind,
    SimulatedResearchSource,
)
from jarvis.web_search import WebSearchHit, WebSearchResponse


def test_research_pipeline_generates_report(jarvis_app, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Jarvis runtime is modular and uses separate services.\nMission control exists for approvals.\n", encoding="utf-8")
    task = jarvis_app.runtime_service.research_run(
        ResearchRunRequest(
            query="Investiga la arquitectura del runtime de Jarvis",
            paths=[str(tmp_path / "notes.md")],
            source_scope=("workspace", "simulated"),
            simulated_sources=[
                SimulatedResearchSource(
                    title="spec",
                    content="Jarvis has autonomy, mission control and modular runtimes.",
                    location="sim://spec",
                )
            ],
        )
    )
    assert task.status.value == "completed"
    assert task.report is not None
    assert task.report.short_summary
    assert task.findings
    assert task.validations


def test_research_can_delegate_to_autonomy(jarvis_app) -> None:
    task = jarvis_app.runtime_service.research_run(
        {
            "query": "Research the runtime architecture",
            "run_via_autonomy": True,
            "source_scope": ["simulated"],
            "simulated_sources": [{"title": "spec", "content": "Jarvis runtime is modular.", "location": "sim://spec"}],
        }
    )
    assert task.mission_id is not None
    assert task.status.value == "delegated"


def test_empty_collection_finishes_with_explicit_memory_stages(jarvis_app) -> None:
    task = jarvis_app.runtime_service.research_run(
        ResearchRunRequest(
            query="coleccion vacia sin evidencia",
            collection_name="missing-collection",
            source_scope=("semantic_memory", "memory"),
            persist_results=False,
        )
    )

    assert task.status.value == "completed"
    assert task.report is not None
    assert task.sources == []
    assert {
        ResearchStepKind.SEMANTIC_MEMORY_RETRIEVAL,
        ResearchStepKind.MEMORY_RETRIEVAL,
    }.issubset({step.kind for step in task.steps})


def test_web_retrieval_is_a_real_attributed_stage(jarvis_app) -> None:
    class FakeWebProvider:
        def search(self, query: str, *, max_results: int = 5):
            assert query == "estado verificable de Nemotron"
            assert max_results > 0
            return WebSearchResponse(
                status="ok",
                provider="test-web",
                query=query,
                hits=[
                    WebSearchHit(
                        title="Fuente primaria",
                        url="https://example.test/nemotron",
                        snippet="Evidencia web atribuida.",
                        source="example.test",
                        rank=1,
                    )
                ],
            )

    retriever = jarvis_app.research_runtime_service._pipeline._retriever  # noqa: SLF001
    retriever._web_provider = FakeWebProvider()  # noqa: SLF001

    task = jarvis_app.runtime_service.research_run(
        ResearchRunRequest(
            query="estado verificable de Nemotron",
            source_scope=("web",),
            persist_results=False,
        )
    )

    assert task.status.value == "completed"
    assert task.sources[0].provider_name == "test-web"
    assert task.evidence[0].citation == "https://example.test/nemotron"
    web_step = next(
        step for step in task.steps if step.kind is ResearchStepKind.WEB_RETRIEVAL
    )
    assert web_step.output_payload["source_count"] == 1


def test_synthesis_failure_preserves_sources_and_returns_partial_report(
    jarvis_app,
) -> None:
    pipeline = jarvis_app.research_runtime_service._pipeline  # noqa: SLF001

    def _fail(*_args, **_kwargs):
        raise RuntimeError("synthetic synthesis failure")

    pipeline._synthesizer.synthesize = _fail  # type: ignore[method-assign]  # noqa: SLF001
    task = jarvis_app.runtime_service.research_run(
        ResearchRunRequest(
            query="resultado parcial",
            source_scope=("simulated",),
            simulated_sources=[
                SimulatedResearchSource(
                    title="evidencia",
                    content="La evidencia debe conservarse aunque falle la sintesis.",
                    location="sim://partial",
                )
            ],
            persist_results=False,
        )
    )

    assert task.status.value == "completed"
    assert task.report is not None
    assert task.report.metadata["partial_results"] is True
    assert task.sources[0].location == "sim://partial"
    synthesis_step = next(
        step for step in task.steps if step.kind is ResearchStepKind.SYNTHESIS
    )
    assert synthesis_step.status.value == "failed"
