from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector
from jarvis.memory_semantic.embeddings import EmbeddingProviderRegistry, EmbeddingRouter, EmbeddingService, build_default_embedding_profiles


class FakeEmbeddingProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.provider_name = name
        self.provider_kind = "local"
        self._fail = fail

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=not self._fail, details={"fake": True})

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        if self._fail:
            raise RuntimeError(f"{self.provider_name} failed")
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=[
                EmbeddingVector(index=index, text=text, values=[float(len(text)), float(index + 1)], dimensions=2)
                for index, text in enumerate(request.texts)
            ],
            latency_ms=2.0,
        )


def test_embedding_router_filters_by_allowed_provider_kind() -> None:
    app = build_application(Settings(ollama_enabled=False, embeddings_enabled=False))
    router = EmbeddingRouter(build_default_embedding_profiles(Settings()), app.mode_manager)
    profiles = router.route(EmbeddingRequest(texts=("hola",), task_type="retrieval"), preferred_provider_order=("ollama_embeddings",))
    assert profiles
    assert all(profile.provider_kind == "local" for profile in profiles)


def test_embedding_service_applies_provider_fallback(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        ollama_enabled=False,
        embeddings_enabled=False,
        embedding_provider_default="primary_embeddings",
        embedding_provider_fallback_order=("primary_embeddings", "secondary_embeddings"),
    )
    app = build_application(settings)
    registry = EmbeddingProviderRegistry()
    registry.register(FakeEmbeddingProvider("primary_embeddings", fail=True))
    registry.register(FakeEmbeddingProvider("secondary_embeddings"))
    router = EmbeddingRouter(
        [
            build_default_embedding_profiles(settings)[0].model_copy(update={"provider": "primary_embeddings", "fallbacks": ("secondary_embedding",), "logical_name": "primary_embedding"}),
            build_default_embedding_profiles(settings)[0].model_copy(update={"provider": "secondary_embeddings", "logical_name": "secondary_embedding", "priority": 2}),
        ],
        app.mode_manager,
    )
    service = EmbeddingService(settings, app.mode_manager, registry, router, app.event_bus)
    response = service.embed(EmbeddingRequest(texts=("alpha",), logical_model="primary_embedding"))
    assert response.provider_name == "secondary_embeddings"
    assert response.fallback_used is True
