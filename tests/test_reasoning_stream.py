from __future__ import annotations

import json

import httpx

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelRequest, StreamChunk
from jarvis.models_runtime.nvidia_nim import NvidiaNimProvider
from jarvis.security_runtime.secrets import EnvironmentSecretStore, NvidiaSecretManager

SECRET = "nvapi-test-key"


def _manager() -> NvidiaSecretManager:
    return NvidiaSecretManager(
        secure_store=None,
        environment_store=EnvironmentSecretStore(environ={"NVIDIA_API_KEY": SECRET}),
    )


def _settings(model: str) -> Settings:
    return Settings(
        _env_file=None,
        nvidia_base_url="https://integrate.api.nvidia.test/v1",
        nvidia_model=model,
    )


def _request(max_tokens: int, *, profile: str | None = "desktop_chat") -> ModelRequest:
    metadata = {"generation_profile": profile} if profile else {}
    return ModelRequest(
        task_type="chat",
        prompt="hola",
        max_tokens=max_tokens,
        metadata=metadata,
    )


def test_stream_chunk_carries_reasoning_text() -> None:
    chunk = StreamChunk(reasoning="Paso uno: revisar el contexto.")

    assert chunk.reasoning == "Paso uno: revisar el contexto."
    assert chunk.text == ""


def test_stream_infer_exposes_reasoning_deltas_to_the_owner() -> None:
    sse = (
        b'data: {"choices":[{"delta":{"reasoning_content":"Primero pienso el plan."}}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning_content":"Luego lo ejecuto."}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"La respuesta es 42"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaNimProvider(
        _settings("zai-org/glm-5.3-xhigh"),
        secret_manager=_manager(),
        client=client,
    )

    chunks = list(
        provider.stream_infer(
            ModelRequest(task_type="chat", prompt="hola"),
            model_name="zai-org/glm-5.3-xhigh",
            temperature=0.2,
            timeout_seconds=5,
        )
    )

    reasoning = "".join(chunk.reasoning for chunk in chunks)
    text = "".join(chunk.text for chunk in chunks)
    assert "Primero pienso el plan." in reasoning
    assert "Luego lo ejecuto." in reasoning
    assert "La respuesta es 42" in text
    assert "pienso" not in text
    assert chunks[-1].done is True


def test_reasoning_budget_auto_applies_to_reasoning_model_families() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "model": "zai-org/glm-5.3-xhigh",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaNimProvider(
        _settings("zai-org/glm-5.3-xhigh"),
        secret_manager=_manager(),
        client=client,
    )

    provider.infer(
        _request(max_tokens=256),
        model_name="zai-org/glm-5.3-xhigh",
        temperature=0.2,
        timeout_seconds=5,
    )

    minimum = provider._reasoning_min_completion_tokens  # noqa: SLF001
    assert captured[0]["max_tokens"] >= minimum


def test_reasoning_budget_respects_explicit_override_and_plain_models() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "model": "meta/llama-plain-8b",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def make_provider(model: str, *, override: bool | None = None) -> NvidiaNimProvider:
        settings = _settings(model)
        if override is not None:
            settings = settings.model_copy(update={"nvidia_reasoning_budget": override})
        return NvidiaNimProvider(settings, secret_manager=_manager(), client=client)

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )

    plain = make_provider("meta/llama-plain-8b")
    plain.infer(
        _request(max_tokens=256),
        model_name="meta/llama-plain-8b",
        temperature=0.2,
        timeout_seconds=5,
    )
    assert captured[0]["max_tokens"] == 256

    captured.clear()
    disabled = make_provider("zai-org/glm-5.3-xhigh", override=False)
    disabled.infer(
        _request(max_tokens=256),
        model_name="zai-org/glm-5.3-xhigh",
        temperature=0.2,
        timeout_seconds=5,
    )
    assert captured[0]["max_tokens"] == 256
