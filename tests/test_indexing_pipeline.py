from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class DeterministicEmbeddingProvider:
    provider_name = "indexing_deterministic"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [
            EmbeddingVector(index=index, text=text, values=[float(len(text.split())), 1.0 if "jarvis" in text.casefold() else 0.0], dimensions=2)
            for index, text in enumerate(request.texts)
        ]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_indexing_pipeline_indexes_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Jarvis\n\nPersistent indexing improves retrieval.", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="indexing_deterministic",
        embedding_provider_fallback_order=("indexing_deterministic",),
        embedding_model_default="indexing-test",
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    app.start()
    try:
        receipt = app.runtime_service.indexing_run({"source_ids": ["workspace"]})
        status = app.runtime_service.indexing_status()
        assert receipt.progress.completed >= 1
        assert status["total_documents"] >= 1
        assert status["total_chunks"] >= 1
        assert any(source["source_id"] == "workspace" for source in status["sources"])
    finally:
        app.stop()
