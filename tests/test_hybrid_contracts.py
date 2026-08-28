from __future__ import annotations

import json

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.models_runtime.hybrid import (
    LLMMode,
    ProviderCapabilities,
    ProviderOperationalState,
    ProviderSelection,
    ProviderStatus,
    RequestExecutionContext,
)
from jarvis.persistent_config import PersistentConfig, load_persistent_config


REMOTE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


def test_hybrid_contracts_keep_mode_provider_and_capabilities_separate() -> None:
    capabilities = ProviderCapabilities(
        text=True,
        images=True,
        audio=True,
        video=True,
        documents=False,
        streaming=True,
        reasoning=True,
        structured_output=True,
        tool_calling=True,
        max_context_tokens=262_144,
        max_files=16,
        input_formats=("text/plain", "image/png", "audio/mpeg", "video/mp4"),
        native_transport_modalities=("image",),
        verification_source="nvidia-model-card",
    )
    status = ProviderStatus(
        provider_name="nvidia_nim",
        state=ProviderOperationalState.CLOUD_HEALTHY,
        healthy=True,
        latency_ms=42.5,
        message="NVIDIA API disponible",
    )
    selection = ProviderSelection(
        selected_mode=LLMMode.AUTO,
        active_provider="nvidia_nim",
        request_provider="nvidia_nim",
        active_model=REMOTE_MODEL,
        provider_status=status,
        capabilities=capabilities,
    )

    assert selection.selected_mode is LLMMode.AUTO
    assert selection.request_provider == selection.active_provider
    assert selection.capabilities.supports_modality("image") is True
    assert selection.capabilities.supports_modality("document") is False
    assert selection.capabilities.supports_native_transport("image") is True
    assert selection.capabilities.supports_native_transport("audio") is False
    assert selection.fallback_reason is None


def test_request_execution_context_prevents_duplicate_files_and_tool_calls() -> None:
    context = RequestExecutionContext(turn_id="turn-1", request_id="request-1")

    assert context.register_file("file-1") is True
    assert context.register_file("file-1") is False
    assert context.register_tool_call("tool-1", external_effect=True) is True
    assert context.register_tool_call("tool-1", external_effect=True) is False
    assert context.has_external_effects is True
    assert context.can_retry_stream(emitted_content=False) is False
    assert context.can_retry_stream(emitted_content=True) is False


def test_settings_accept_required_unprefixed_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_LLM_MODE", "local")
    monkeypatch.setenv("NVIDIA_API_KEY", "sentinel-secret-value")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("NVIDIA_MODEL", REMOTE_MODEL)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://local-host:9000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", LOCAL_MODEL)
    monkeypatch.setenv("LOCAL_CPU_OFFLOAD_GB", "12")
    monkeypatch.setenv("LOCAL_GPU_MEMORY_UTILIZATION", "0.82")
    monkeypatch.setenv("LOCAL_MAX_MODEL_LEN", "32768")
    monkeypatch.setenv("LOCAL_TENSOR_PARALLEL_SIZE", "2")

    settings = Settings(_env_file=None)

    assert settings.llm_mode == "local"
    assert settings.nvidia_base_url == "https://example.invalid/v1"
    assert settings.nvidia_model == REMOTE_MODEL
    assert settings.local_llm_base_url == "http://local-host:9000/v1"
    assert settings.local_llm_model == LOCAL_MODEL
    assert settings.local_cpu_offload_gb == "12"
    assert settings.local_gpu_memory_utilization == "0.82"
    assert settings.local_max_model_len == "32768"
    assert settings.local_tensor_parallel_size == "2"
    assert "sentinel-secret-value" not in repr(settings)
    assert "sentinel-secret-value" not in json.dumps(settings.model_dump(), default=str)


def test_settings_default_to_auto_and_exact_nemotron_models(monkeypatch) -> None:
    for name in (
        "JARVIS_LLM_MODE",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "NVIDIA_MODEL",
        "LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_mode == "auto"
    assert settings.nvidia_model == REMOTE_MODEL
    assert settings.local_llm_model == LOCAL_MODEL


def test_persistent_config_migrates_legacy_mode_and_provider_to_auto_nemotron(
    tmp_path,
) -> None:
    legacy_provider = "oll" + "ama"
    legacy_model = "gpt" + "-oss:20b"
    config_path = tmp_path / "runtime" / "jarvis_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "preferred_mode": "online",
                "local_provider": legacy_provider,
                "local_base_url": "http://127.0.0.1:11434",
                "local_model": legacy_model,
            }
        ),
        encoding="utf-8",
    )

    config = load_persistent_config(tmp_path)

    assert config.preferred_mode == "auto"
    assert config.local_provider == "nemotron_omni_local"
    assert config.local_base_url == "http://127.0.0.1:8000/v1"
    assert config.local_model == LOCAL_MODEL


def test_persistent_config_defaults_target_nemotron_local() -> None:
    config = PersistentConfig.defaults()

    assert config.preferred_mode == "auto"
    assert config.local_provider == "nemotron_omni_local"
    assert config.local_base_url == "http://127.0.0.1:8000/v1"
    assert config.local_model == LOCAL_MODEL


@pytest.mark.parametrize("mode", ["auto", "nvidia_api", "local"])
def test_application_always_uses_provider_coordinator_for_supported_modes(
    tmp_path, mode: str, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "jarvis.persistent_config.user_config_dir",
        lambda: tmp_path / "user-config",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode=mode,
        embeddings_enabled=False,
    )

    app = build_application(settings)

    assert app.model_service._coordinator is not None  # noqa: SLF001
    assert app.model_service.provider_selection().selected_mode.value == mode
