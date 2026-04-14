from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class DeterministicEmbeddingProvider:
    provider_name = "indexing_incremental_provider"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [EmbeddingVector(index=index, text=text, values=[float(len(text)), 1.0], dimensions=2) for index, text in enumerate(request.texts)]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_indexing_incremental_skips_unchanged_and_updates_changed(tmp_path: Path) -> None:
    target = tmp_path / "draft.md"
    target.write_text("Jarvis draft one", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="indexing_incremental_provider",
        embedding_provider_fallback_order=("indexing_incremental_provider",),
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    app.start()
    try:
        first = app.runtime_service.indexing_run({"source_ids": ["workspace"]})
        second = app.runtime_service.indexing_run({"source_ids": ["workspace"]})
        target.write_text("Jarvis draft two updated", encoding="utf-8")
        third = app.runtime_service.indexing_run({"source_ids": ["workspace"]})
        docs = app.indexing_runtime_service._repository.list_documents("workspace")  # noqa: SLF001
        assert first.progress.completed >= 1
        assert second.progress.skipped >= 1
        assert third.progress.completed >= 1
        assert len(docs) == 1
        assert "updated" in docs[0].content
    finally:
        app.stop()
