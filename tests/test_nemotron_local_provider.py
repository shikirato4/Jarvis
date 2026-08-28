from __future__ import annotations

import json

import httpx
import pytest

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest
from jarvis.models_runtime.hybrid import ProviderOperationalState
from jarvis.models_runtime.nemotron_local import (
    LocalModelMismatchError,
    NemotronOmniLocalProvider,
)


LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        local_llm_base_url="http://local-host:8000/v1",
        local_llm_model=LOCAL_MODEL,
    )


def test_local_provider_builds_plain_http_transport_without_ssl_store(monkeypatch) -> None:
    captured: dict = {}
    sentinel = object()

    def _client(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "jarvis.models_runtime.openai_compatible.httpx.Client",
        _client,
    )
    provider = NemotronOmniLocalProvider(_settings())

    assert provider._http_client() is sentinel  # noqa: SLF001
    assert captured["verify"] is False
    assert captured["trust_env"] is False


def test_local_provider_uses_only_configured_openai_compatible_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": LOCAL_MODEL}]})
        payload = json.loads(request.content)
        assert payload["model"] == LOCAL_MODEL
        return httpx.Response(
            200,
            json={"model": LOCAL_MODEL, "choices": [{"message": {"content": "local"}}]},
        )

    client = httpx.Client(
        base_url="http://local-host:8000/v1", transport=httpx.MockTransport(handler)
    )
    provider = NemotronOmniLocalProvider(_settings(), client=client)

    status = provider.status_check()
    response = provider.infer(
        ModelRequest(task_type="assistant", prompt="hola"),
        model_name=LOCAL_MODEL,
        temperature=0.2,
        timeout_seconds=3,
    )

    assert status.state is ProviderOperationalState.LOCAL_ACTIVE
    assert response.content == "local"
    assert all(request.url.host == "local-host" for request in requests)
    assert all("Authorization" not in request.headers for request in requests)


def test_local_provider_never_changes_model_or_quantization_silently() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    provider = NemotronOmniLocalProvider(
        _settings(),
        client=httpx.Client(
            base_url="http://local-host:8000/v1", transport=httpx.MockTransport(handler)
        ),
    )

    with pytest.raises(LocalModelMismatchError, match="NVFP4"):
        provider.infer(
            ModelRequest(task_type="assistant", prompt="hola"),
            model_name="different-model",
            temperature=None,
            timeout_seconds=2,
        )

    assert calls == 0


def test_local_provider_stopped_returns_exact_start_diagnostic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = NemotronOmniLocalProvider(
        _settings(),
        client=httpx.Client(
            base_url="http://local-host:8000/v1", transport=httpx.MockTransport(handler)
        ),
    )

    status = provider.status_check()

    assert status.healthy is False
    assert status.state is ProviderOperationalState.LOCAL_UNAVAILABLE
    assert "http://local-host:8000/v1" in (status.message or "")
    assert "inicia" in (status.message or "").casefold()
    assert LOCAL_MODEL in (status.message or "")


def test_local_health_rejects_endpoint_serving_a_different_model() -> None:
    provider = NemotronOmniLocalProvider(
        _settings(),
        client=httpx.Client(
            base_url="http://local-host:8000/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"data": [{"id": "different-model"}]}
                )
            ),
        ),
    )

    status = provider.status_check()

    assert status.healthy is False
    assert status.state is ProviderOperationalState.LOCAL_UNAVAILABLE
    assert LOCAL_MODEL in (status.message or "")
    assert "no se cambiará silenciosamente" in (status.message or "")


def test_local_capabilities_reflect_model_card_but_limit_verified_native_transport() -> (
    None
):
    provider = NemotronOmniLocalProvider(
        _settings(),
        client=httpx.Client(
            base_url="http://local-host:8000/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"data": [{"id": LOCAL_MODEL}]}
                )
            ),
        ),
    )

    assert provider.capabilities.text is True
    assert provider.capabilities.streaming is True
    assert provider.capabilities.reasoning is True
    assert provider.capabilities.images is True
    assert provider.capabilities.audio is True
    assert provider.capabilities.video is True
    assert provider.capabilities.structured_output is True
    assert provider.capabilities.tool_calling is True
    assert provider.capabilities.max_context_tokens == 262_144
    assert provider.capabilities.native_transport_modalities == ("image",)
    assert "official-nvidia-model-card" in (
        provider.capabilities.verification_source or ""
    )
