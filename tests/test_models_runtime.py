from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.base import (
    ModelRequest,
    ModelResponse,
    ProviderHealth,
    ProviderKind,
    StreamChunk,
)
from jarvis.models_runtime.catalog import build_default_model_catalog
from jarvis.models_runtime.hybrid import ProviderCapabilities, ProviderOperationalState, ProviderStatus


LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


class FakeLocalProvider:
    provider_name = "qwen_local"
    provider_kind = ProviderKind.LOCAL
    configured_model = LOCAL_MODEL
    capabilities = ProviderCapabilities(text=True, streaming=True, reasoning=True)

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(provider_name=self.provider_name, healthy=True)

    def status_check(self) -> ProviderStatus:
        return ProviderStatus(
            provider_name=self.provider_name,
            state=ProviderOperationalState.LOCAL_ACTIVE,
            healthy=True,
            message="fake local ready",
        )

    def infer(self, request, *, model_name, temperature, timeout_seconds) -> ModelResponse:
        return ModelResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or "general_assistant",
            model_name=model_name,
            content="respuesta local",
            latency_ms=1,
        )

    def stream_infer(
        self,
        request,
        *,
        model_name,
        temperature,
        timeout_seconds,
        cancel_check=None,
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(text="respuesta ")
        yield StreamChunk(text="local")
        yield StreamChunk(done=True)


def _application(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="local",
        embeddings_enabled=False,
    )
    application = build_application(settings)
    fake = FakeLocalProvider()
    application.provider_registry._providers.pop("qwen_local", None)  # noqa: SLF001
    application.provider_registry._providers["qwen_local"] = fake  # noqa: SLF001
    application.model_service._coordinator._local = fake  # noqa: SLF001
    application.model_service._coordinator._selected_mode = application.model_service.provider_selection().selected_mode.LOCAL  # noqa: SLF001
    return application


def test_build_application_registers_online_and_qwen_local_providers(tmp_path: Path) -> None:
    application = _application(tmp_path)

    assert [
        provider.provider_name
        for provider in application.provider_registry.list_providers()
    ] == ["nvidia_nim", "qwen_local"]


def test_model_catalog_keeps_capabilities_on_coordinator_profiles() -> None:
    catalog = build_default_model_catalog(Settings(_env_file=None))

    reasoning = catalog.get("reasoning_engine")
    coding = catalog.get("coding_engine")
    assert reasoning is not None and "planning" in reasoning.capabilities
    assert coding is not None and "coding" in coding.capabilities
    assert all(
        profile.provider == "provider_coordinator"
        for profile in catalog.list_profiles()
    )


def test_model_service_routes_inference_through_required_coordinator(tmp_path: Path) -> None:
    application = _application(tmp_path)

    response = application.model_service.infer(
        ModelRequest(task_type="assistant", prompt="hola")
    )

    assert response.provider_name == "qwen_local"
    assert response.model_name == LOCAL_MODEL
    assert response.content == "respuesta local"


def test_model_service_streams_through_required_coordinator(tmp_path: Path) -> None:
    application = _application(tmp_path)

    chunks = list(
        application.model_service.stream(
            ModelRequest(task_type="assistant", prompt="hola")
        )
    )

    assert "".join(chunk.text for chunk in chunks) == "respuesta local"
    assert chunks[-1].metadata["active_provider"] == "qwen_local"


def test_local_hash_embeddings_preserve_semantic_memory_without_network(tmp_path: Path) -> None:
    application = build_application(
        Settings(
            _env_file=None,
            data_dir=tmp_path / "runtime",
            workspace_root=tmp_path,
            llm_mode="local",
        )
    )

    providers = application.embedding_provider_registry.list_providers()
    assert [provider.provider_name for provider in providers] == ["local_hash_embeddings"]
    assert providers[0].health_check().details["network"] is False
