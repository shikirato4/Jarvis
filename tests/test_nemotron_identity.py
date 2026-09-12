from __future__ import annotations

import json

import httpx
import pytest

from jarvis.config import Settings
from jarvis.legacy_model_config import detect_legacy_model_configuration
from jarvis.models_runtime.base import ModelRequest
from jarvis.models_runtime.nemotron_identity import (
    InvalidNemotronIdentityError,
    NemotronIdentity,
    local_model_policy_description,
)
from jarvis.models_runtime.nemotron_local import NemotronOmniLocalProvider
from jarvis.persistent_config import (
    PersistentConfig,
    _config_from_payload,
    _payload_from_config,
)


CANONICAL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
GGUF = "lmstudio-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-Q4_K_M.gguf"
QWEN_GGUF = "Qwen3.5-9B-IQ4_XS"


@pytest.mark.parametrize(
    "advertised",
    [
        CANONICAL,
        GGUF,
        "Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4",
        "nemotron-3-nano-omni-30b-a3b-reasoning",
    ],
)
def test_nemotron_identity_accepts_valid_runtime_variants(advertised: str) -> None:
    identity = NemotronIdentity(configured_model=CANONICAL)

    assert identity.matches(advertised) is True
    assert identity.select_advertised(["unrelated-model", advertised]) == advertised


def test_nemotron_identity_accepts_only_explicit_opaque_aliases() -> None:
    identity = NemotronIdentity(
        configured_model=CANONICAL,
        aliases=("lm-studio-loaded-model",),
    )

    assert identity.matches("lm-studio-loaded-model") is True
    assert identity.matches("some-other-loaded-model") is False


@pytest.mark.parametrize(
    "invalid",
    [
        "meta-llama/Llama-3.1-8B-Instruct",
        "nvidia/Nemotron-Mini-4B-Instruct",
        "nvidia/Nemotron-3-Nano-Omni-8B-Reasoning",
        "",
    ],
)
def test_nemotron_identity_rejects_other_model_families(invalid: str) -> None:
    with pytest.raises(InvalidNemotronIdentityError):
        NemotronIdentity(configured_model=invalid)


def test_local_provider_resolves_gguf_advertised_id_for_inference() -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": GGUF}]})
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": GGUF, "choices": [{"message": {"content": "local"}}]},
        )

    settings = Settings(
        _env_file=None,
        local_llm_base_url="http://127.0.0.1:1234/v1",
        local_llm_model=CANONICAL,
    )
    provider = NemotronOmniLocalProvider(
        settings,
        client=httpx.Client(
            base_url=settings.local_llm_base_url,
            transport=httpx.MockTransport(handler),
        ),
    )

    assert provider.status_check().healthy is True
    response = provider.infer(
        ModelRequest(task_type="assistant", prompt="hola"),
        model_name=provider.configured_model,
        temperature=None,
        timeout_seconds=2,
    )

    assert provider.configured_model == GGUF
    assert payloads[0]["model"] == GGUF
    assert response.model_name == GGUF


def test_persistent_config_preserves_valid_local_url_model_and_aliases() -> None:
    config = PersistentConfig(
        preferred_mode="local",
        local_base_url="http://127.0.0.1:1234/v1",
        local_model=QWEN_GGUF,
        local_model_aliases=("lm-studio-loaded-model",),
    )

    payload = _payload_from_config(config)
    restored = _config_from_payload(payload)

    assert restored.preferred_mode == "local"
    assert restored.local_base_url == "http://127.0.0.1:1234/v1"
    assert restored.local_model == QWEN_GGUF
    assert restored.local_model_aliases == ("lm-studio-loaded-model",)


def test_legacy_model_configuration_diagnostic_exposes_names_not_values() -> None:
    diagnostic = detect_legacy_model_configuration(
        environ={
            "GPT_OSS_MODEL": "private-old-model-value",
            "OLLAMA_HOST": "private-old-host-value",
            "NVIDIA_API_KEY": "private-current-secret",
        },
        env_file=None,
        emit_log=False,
    )

    assert diagnostic.obsolete_keys == ("GPT_OSS_MODEL", "OLLAMA_HOST")
    assert diagnostic.ignored is True
    rendered = repr(diagnostic)
    assert "private-old-model-value" not in rendered
    assert "private-old-host-value" not in rendered
    assert "private-current-secret" not in rendered


def test_legacy_environment_cannot_select_gpt_oss(monkeypatch) -> None:
    monkeypatch.setenv("GPT_OSS_MODEL", "old-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.delenv("JARVIS_LLM_MODE", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_mode == "auto"
    assert "qwen" in settings.local_llm_model.casefold()
    assert "gpt" + "-oss" not in settings.local_llm_model.casefold()


def test_local_policy_diagnostic_describes_backend_neutral_nemotron_identity() -> None:
    description = local_model_policy_description()

    assert "Nemotron 3 Nano Omni 30B A3B Reasoning" in description
    assert "NVFP4/GGUF" in description
    assert "alias" in description.casefold()
    assert "strict Nemotron NVFP4 endpoint" not in description
