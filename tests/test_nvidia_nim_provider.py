from __future__ import annotations

import json

import httpx
import pytest

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMedia, ModelRequest
from jarvis.models_runtime.hybrid import ProviderOperationalState
from jarvis.models_runtime.nvidia_nim import (
    NvidiaInvalidKeyError,
    NvidiaNimProvider,
    NvidiaRateLimitError,
    NvidiaUnavailableError,
)
from jarvis.security_runtime.secrets import EnvironmentSecretStore, NvidiaSecretManager


REMOTE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
SECRET = "nvapi-sentinel-never-log"


def _manager(value: str | None) -> NvidiaSecretManager:
    environ = {"NVIDIA_API_KEY": value} if value else {}
    return NvidiaSecretManager(
        secure_store=None,
        environment_store=EnvironmentSecretStore(environ=environ),
    )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        nvidia_base_url="https://integrate.api.nvidia.test/v1",
        nvidia_model=REMOTE_MODEL,
    )


def test_nvidia_provider_defers_http_transport_creation(monkeypatch) -> None:
    calls = 0

    def _unexpected_client(*args, **kwargs):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP transport must be lazy during provider construction")

    monkeypatch.setattr(
        "jarvis.models_runtime.openai_compatible.httpx.Client",
        _unexpected_client,
    )

    provider = NvidiaNimProvider(_settings(), secret_manager=_manager(SECRET))

    assert provider.base_url == "https://integrate.api.nvidia.test/v1"
    assert calls == 0


def test_nvidia_health_is_authenticated_and_capabilities_are_explicit() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"id": REMOTE_MODEL}]})

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    manager = _manager(SECRET)
    provider = NvidiaNimProvider(_settings(), secret_manager=manager, client=client)

    status = provider.status_check()

    assert status.healthy is True
    assert status.state is ProviderOperationalState.CLOUD_HEALTHY
    assert requests[0].url.path == "/v1/models"
    assert requests[0].headers["Authorization"] == f"Bearer {SECRET}"
    assert provider.capabilities.images is True
    assert provider.capabilities.audio is True
    assert provider.capabilities.video is True
    assert provider.capabilities.documents is False
    assert provider.capabilities.tool_calling is True
    assert provider.capabilities.max_context_tokens == 262_144
    assert manager.status().validity == "valid"


def test_nvidia_missing_key_does_not_make_a_network_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(None), client=client
    )

    status = provider.status_check()

    assert calls == 0
    assert status.healthy is False
    assert status.state is ProviderOperationalState.CLOUD_INVALID_KEY
    assert "no configurada" in (status.message or "").casefold()


def test_nvidia_rejected_key_updates_only_redacted_validity_status() -> None:
    manager = _manager(SECRET)
    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=manager,
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(lambda request: httpx.Response(403)),
        ),
    )

    status = provider.status_check()
    key_status = manager.status().model_dump(mode="json")

    assert status.state is ProviderOperationalState.CLOUD_INVALID_KEY
    assert key_status == {
        "configured": True,
        "source": "environment",
        "validity": "invalid",
    }
    assert SECRET not in json.dumps(key_status)


def test_nvidia_inference_uses_exact_model_and_normalizes_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": REMOTE_MODEL,
                "choices": [{"message": {"role": "assistant", "content": "Hola"}}],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            },
        )

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(SECRET), client=client
    )
    response = provider.infer(
        ModelRequest(
            task_type="assistant",
            messages=[{"role": "user", "content": "Hola"}],
            max_tokens=20,
        ),
        model_name=REMOTE_MODEL,
        temperature=0.25,
        timeout_seconds=3,
    )

    assert captured["model"] == REMOTE_MODEL
    assert captured["messages"] == [{"role": "user", "content": "Hola"}]
    assert captured["max_tokens"] == 20
    assert response.content == "Hola"
    assert response.provider_name == "nvidia_nim"
    assert response.usage.total_tokens == 6


def test_nemotron_chat_profile_reserves_tokens_for_visible_answer() -> None:
    settings = _settings().model_copy(
        update={"nvidia_reasoning_min_completion_tokens": 512}
    )
    provider = NvidiaNimProvider(settings, secret_manager=_manager(SECRET))

    payload = provider._request_payload(  # noqa: SLF001
        ModelRequest(
            task_type="assistant",
            prompt="hola",
            max_tokens=160,
            metadata={"generation_profile": "fast"},
        ),
        temperature=0.2,
        stream=True,
    )
    direct_payload = provider._request_payload(  # noqa: SLF001
        ModelRequest(task_type="assistant", prompt="hola", max_tokens=20),
        temperature=0.2,
        stream=False,
    )

    assert payload["max_tokens"] == 512
    assert direct_payload["max_tokens"] == 20


def test_nvidia_tool_calling_and_structured_output_are_forwarded_and_normalized() -> (
    None
):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": REMOTE_MODEL,
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_status",
                                        "arguments": '{"scope":"local"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=_manager(SECRET),
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    response = provider.infer(
        ModelRequest(
            task_type="assistant",
            prompt="estado",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_status",
                        "description": "Read local status",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
            response_format={"type": "json_object"},
        ),
        model_name=REMOTE_MODEL,
        temperature=0.1,
        timeout_seconds=2,
    )

    assert captured["tools"][0]["function"]["name"] == "read_status"
    assert captured["tool_choice"] == "auto"
    assert captured["response_format"] == {"type": "json_object"}
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].name == "read_status"
    assert response.tool_calls[0].arguments == {"scope": "local"}


def test_nvidia_encodes_dotted_terminal_actions_and_restores_canonical_call() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": REMOTE_MODEL,
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "terminal-1",
                            "type": "function",
                            "function": {
                                "name": "terminal__run",
                                "arguments": '{"session_id":"s","command":"echo ok"}',
                            },
                        }],
                    }
                }],
            },
        )

    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=_manager(SECRET),
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    response = provider.infer(
        ModelRequest(
            task_type="agent",
            prompt="usa terminal",
            tools=[{
                "type": "function",
                "function": {
                    "name": "terminal.run",
                    "description": "Run a terminal command",
                    "parameters": {"type": "object"},
                },
            }],
        ),
        model_name=REMOTE_MODEL,
        temperature=0.1,
        timeout_seconds=2,
    )

    assert captured["tools"][0]["function"]["name"] == "terminal__run"
    assert response.tool_calls[0].name == "terminal.run"


def test_nvidia_encodes_and_restores_dotted_file_action() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": REMOTE_MODEL,
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [{
                            "id": "files-1",
                            "type": "function",
                            "function": {
                                "name": "files__search",
                                "arguments": '{"pattern":"report.pdf"}',
                            },
                        }],
                    }
                }],
            },
        )

    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=_manager(SECRET),
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    response = provider.infer(
        ModelRequest(
            task_type="agent",
            prompt="busca report.pdf",
            tools=[{
                "type": "function",
                "function": {
                    "name": "files.search",
                    "description": "Search for a file",
                    "parameters": {"type": "object"},
                },
            }],
        ),
        model_name=REMOTE_MODEL,
        temperature=0.1,
        timeout_seconds=2,
    )
    assert captured["tools"][0]["function"]["name"] == "files__search"
    assert response.tool_calls[0].name == "files.search"


@pytest.mark.parametrize("status_code", [401, 403])
def test_nvidia_invalid_key_errors_are_typed_and_redacted(status_code: int) -> None:
    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status_code, text=SECRET)
        ),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(SECRET), client=client
    )

    with pytest.raises(NvidiaInvalidKeyError) as captured:
        provider.infer(
            ModelRequest(task_type="assistant", prompt="hola"),
            model_name=REMOTE_MODEL,
            temperature=0.1,
            timeout_seconds=2,
        )

    assert SECRET not in str(captured.value)
    assert SECRET not in repr(captured.value)


def test_nvidia_rate_limit_preserves_retry_after_without_response_body() -> None:
    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                429, headers={"Retry-After": "17"}, text=SECRET
            )
        ),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(SECRET), client=client
    )

    with pytest.raises(NvidiaRateLimitError) as captured:
        provider.infer(
            ModelRequest(task_type="assistant", prompt="hola"),
            model_name=REMOTE_MODEL,
            temperature=None,
            timeout_seconds=2,
        )

    assert captured.value.retry_after_seconds == 17
    assert SECRET not in str(captured.value)


@pytest.mark.parametrize("failure", [500, 503, "timeout"])
def test_nvidia_transient_failures_are_typed_as_unavailable(failure: int | str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("network timeout with no secret", request=request)
        return httpx.Response(int(failure))

    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(SECRET), client=client
    )

    with pytest.raises(NvidiaUnavailableError):
        provider.infer(
            ModelRequest(task_type="assistant", prompt="hola"),
            model_name=REMOTE_MODEL,
            temperature=None,
            timeout_seconds=2,
        )


def test_nvidia_streaming_normalizes_sse_chunks() -> None:
    body = (
        b'data: {"choices":[{"delta":{"content":"Ho"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"la"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    client = httpx.Client(
        base_url="https://integrate.api.nvidia.test/v1",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=body)
        ),
    )
    provider = NvidiaNimProvider(
        _settings(), secret_manager=_manager(SECRET), client=client
    )

    chunks = list(
        provider.stream_infer(
            ModelRequest(task_type="assistant", prompt="hola", stream=True),
            model_name=REMOTE_MODEL,
            temperature=0.2,
            timeout_seconds=2,
        )
    )

    assert "".join(chunk.text for chunk in chunks) == "Hola"
    assert chunks[-1].done is True


def test_nvidia_reasoning_only_stream_is_not_reported_as_completed() -> None:
    private_reasoning = "private chain must never become visible"
    body = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"reasoning_content": private_reasoning}}]}
        )
        + "\n\ndata: [DONE]\n\n"
    ).encode()
    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=_manager(SECRET),
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=body)
            ),
        ),
    )

    with pytest.raises(NvidiaUnavailableError) as captured:
        list(
            provider.stream_infer(
                ModelRequest(task_type="assistant", prompt="hola", stream=True),
                model_name=REMOTE_MODEL,
                temperature=0.2,
                timeout_seconds=2,
            )
        )

    assert private_reasoning not in str(captured.value)


def test_nvidia_image_transport_is_ephemeral_and_not_serialized_in_request_history() -> (
    None
):
    captured: dict = {}
    image_bytes = b"\x89PNG\r\n\x1a\nprivate-image-bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": REMOTE_MODEL, "choices": [{"message": {"content": "ok"}}]},
        )

    provider = NvidiaNimProvider(
        _settings(),
        secret_manager=_manager(SECRET),
        client=httpx.Client(
            base_url="https://integrate.api.nvidia.test/v1",
            transport=httpx.MockTransport(handler),
        ),
    )
    model_request = ModelRequest(
        task_type="assistant",
        prompt="Describe la imagen",
        media=[
            ModelMedia(
                file_id="file-1",
                category="image",
                mime_type="image/png",
                safe_name="image.png",
                data=image_bytes,
            )
        ],
    )

    provider.infer(
        model_request, model_name=REMOTE_MODEL, temperature=0.2, timeout_seconds=2
    )

    assert "media" not in model_request.model_dump(mode="json")
    assert "private-image-bytes" not in repr(model_request)
    content = captured["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe la imagen"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
