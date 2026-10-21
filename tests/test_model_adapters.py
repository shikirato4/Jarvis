from __future__ import annotations

import httpx

from jarvis.models_runtime import (
    LlamaCppAdapter,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ProviderCapabilities,
    VLLMAdapter,
)
from jarvis.models_runtime.base import ModelMessage, ModelRequest


def _client() -> httpx.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "ornith-1.5-9b"}]})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "model": "ornith-1.5-9b",
                "choices": [{"message": {"role": "assistant", "content": "hola"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            })
        return httpx.Response(404)

    return httpx.Client(base_url="http://127.0.0.1/v1", transport=httpx.MockTransport(respond))


def test_local_adapters_share_chat_model_discovery_and_capability_contract() -> None:
    adapters = [
        OllamaAdapter(model="ornith-1.5-9b", client=_client()),
        LlamaCppAdapter(model="ornith-1.5-9b", client=_client()),
        VLLMAdapter(model="ornith-1.5-9b", client=_client()),
        OpenAICompatibleAdapter(
            provider_name="test_endpoint",
            base_url="http://127.0.0.1:1234/v1",
            model="ornith-1.5-9b",
            client=_client(),
        ),
    ]
    for adapter in adapters:
        assert adapter.list_models() == ["ornith-1.5-9b"]
        assert adapter.supports("chat")
        assert adapter.supports("streaming")
        assert not adapter.supports("vision")
        assert adapter.get_model_info()["model"] == "ornith-1.5-9b"
        response = adapter.infer(
            ModelRequest(task_type="assistant", messages=[ModelMessage(role="user", content="hola")]),
            model_name="ornith-1.5-9b",
            temperature=0,
            timeout_seconds=1,
        )
        assert response.content == "hola"
        assert response.provider_name == adapter.provider_name


def test_capabilities_are_operator_configurable_and_conservative_by_default() -> None:
    adapter = OllamaAdapter(
        model="vision-model",
        client=_client(),
        capabilities=ProviderCapabilities(
            text=True,
            images=True,
            streaming=True,
            tool_calling=True,
            verification_source="operator configuration",
        ),
    )
    assert adapter.supports("vision")
    assert adapter.supports("tool_calling")
