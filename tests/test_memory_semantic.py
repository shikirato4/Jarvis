from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.cognition.models import OrchestrationRequest
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector
from jarvis.memory_semantic.documents import DocumentIngestionRequest


class DeterministicEmbeddingProvider:
    provider_name = "semantic_deterministic"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = []
        for index, text in enumerate(request.texts):
            lowered = text.casefold()
            vectors.append(
                EmbeddingVector(
                    index=index,
                    text=text,
                    values=[1.0 if "jarvis" in lowered else 0.0, 1.0 if "contexto" in lowered or "context" in lowered else 0.0],
                    dimensions=2,
                )
            )
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_semantic_ingestion_preserves_operational_memory_contract(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="semantic_deterministic",
        embedding_provider_fallback_order=("semantic_deterministic",),
        embedding_model_default="semantic-test",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    app.start()
    try:
        document = app.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name="library",
                source_type="markdown",
                content="Jarvis mantiene contexto de largo plazo para escritura asistida.",
                title="Long-term context",
                persist_memory=True,
            )
        )
        semantic_result = app.runtime_service.semantic_search({"query": "contexto jarvis", "collection_name": "library"})
        operational_matches = app.memory_service.search_memories("Long-term context", limit=10)
        assert document.collection_name == "library"
        assert semantic_result.context.chunks
        assert operational_matches
        assert all(entry.kind == "semantic.document" for entry in operational_matches)
    finally:
        app.stop()


def test_orchestrator_uses_retrieved_context_for_contextual_writing(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="semantic_deterministic",
        embedding_provider_fallback_order=("semantic_deterministic",),
        embedding_model_default="semantic-test",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    app.start()
    try:
        app.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name="drafts",
                source_type="draft",
                content="Jarvis conserva contexto narrativo para continuidad de borradores.",
                title="Narrative continuity",
            )
        )
        response = app.orchestrator.handle(
            OrchestrationRequest(
                intent="contextual_writing",
                query="Redacta una nota con contexto para Jarvis",
                payload={"title": "Context draft", "objective": "Validar escritura contextual"},
            )
        )
        snapshot = app.runtime_service.snapshot()
        content = response.receipts[0].data["content"]
        assert "continuidad de borradores" in content.casefold()
        assert snapshot.recent_embedding_invocations
        assert snapshot.recent_embedding_invocations[0].provider == "semantic_deterministic"
    finally:
        app.stop()
