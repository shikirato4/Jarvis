from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class DeterministicEmbeddingProvider:
    provider_name = "indexing_api_provider"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [EmbeddingVector(index=index, text=text, values=[1.0, 0.0], dimensions=2) for index, text in enumerate(request.texts)]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_indexing_api_routes(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "api.txt").write_text("Jarvis API indexing", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="indexing_api_provider",
        embedding_provider_fallback_order=("indexing_api_provider",),
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    test_app.embedding_provider_registry.register(DeterministicEmbeddingProvider())
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        status = client.get("/index/status")
        assert status.status_code == 200
        run = client.post("/index/run", json={"source_ids": ["workspace"]})
        assert run.status_code == 200
        add_source = client.post(
            "/index/source",
            json={"source_id": "docs", "source_kind": "user_documents", "display_name": "Docs", "root_path": str(tmp_path)},
        )
        assert add_source.status_code == 200
        reindex = client.post("/index/reindex", json={"source_ids": ["workspace"], "force_reindex": True})
        assert reindex.status_code == 200
