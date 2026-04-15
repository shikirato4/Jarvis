from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class DeterministicEmbeddingProvider:
    provider_name = "indexing_storage_provider"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [EmbeddingVector(index=index, text=text, values=[1.0, float(index)], dimensions=2) for index, text in enumerate(request.texts)]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_indexing_storage_persists_documents_and_chunks(tmp_path: Path) -> None:
    (tmp_path / "doc.txt").write_text("Jarvis indexing storage test", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="indexing_storage_provider",
        embedding_provider_fallback_order=("indexing_storage_provider",),
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    app.start()
    try:
        app.runtime_service.indexing_run({"source_ids": ["workspace"]})
        documents = app.indexing_runtime_service._repository.list_documents("workspace")  # noqa: SLF001
        chunks = app.indexing_runtime_service._repository.list_chunks(documents[0].document_id)  # noqa: SLF001
        assert documents
        assert chunks
        assert chunks[0].embedding_vector
    finally:
        app.stop()
