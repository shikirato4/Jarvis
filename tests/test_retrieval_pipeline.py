from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector, SemanticSearchQuery
from jarvis.memory_semantic.documents import ChunkRecord, CollectionRecord, DocumentIngestionRequest, DocumentProvenance, DocumentRecord


class SemanticFakeProvider:
    provider_name = "semantic_fake"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors: list[EmbeddingVector] = []
        for index, text in enumerate(request.texts):
            lowered = text.casefold()
            vectors.append(
                EmbeddingVector(
                    index=index,
                    text=text,
                    values=[1.0 if "alpha" in lowered else 0.0, 1.0 if "beta" in lowered else 0.0],
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


def test_semantic_retrieval_filters_by_collection_and_source_type(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="semantic_fake",
        embedding_provider_fallback_order=("semantic_fake",),
        embedding_model_default="semantic-test",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(SemanticFakeProvider())
    app.start()
    try:
        app.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name="research",
                source_type="markdown",
                content="alpha evidence in research corpus",
                title="Alpha research",
                metadata={"topic": "alpha"},
            )
        )
        app.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name="notes",
                source_type="note",
                content="beta note in personal corpus",
                title="Beta note",
                metadata={"topic": "beta"},
            )
        )
        result = app.runtime_service.semantic_search(
            SemanticSearchQuery(query="alpha", collection_name="research", source_types=("markdown",))
        )
        assert result.context.chunks
        assert all(chunk.collection_name == "research" for chunk in result.context.chunks)
        assert all(chunk.source_type == "markdown" for chunk in result.context.chunks)
    finally:
        app.stop()


def test_retrieval_degrades_to_lexical_search_without_embedding_provider(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
    )
    app = build_application(settings)
    app.start()
    try:
        repository = app.semantic_memory_service._repository  # noqa: SLF001
        repository.upsert_collection(CollectionRecord(name="fallback"))
        document = repository.save_document(
            DocumentRecord(
                id="doc-fallback",
                collection_name="fallback",
                title="Fallback note",
                source_type="note",
                content="alpha lexical only evidence",
                provenance=DocumentProvenance(source_path=str(tmp_path / "fallback.md")),
            )
        )
        repository.replace_chunks(
            document.id,
            [
                ChunkRecord(
                    id="chunk-fallback",
                    document_id=document.id,
                    collection_name="fallback",
                    chunk_index=0,
                    content="alpha lexical only evidence",
                    source_type="note",
                    provenance=DocumentProvenance(source_path=str(tmp_path / "fallback.md")),
                )
            ],
        )
        result = app.runtime_service.semantic_search(SemanticSearchQuery(query="alpha", collection_name="fallback"))
        assert result.context.degraded is True
        assert result.context.fallback_applied is True
        assert result.context.strategy == "lexical_fallback"
        assert result.context.chunks
    finally:
        app.stop()
