from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest, ModelToolCall
from jarvis.models_runtime.nemotron_local import NemotronOmniLocalProvider


LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._send_json({"object": "list", "data": [{"id": LOCAL_MODEL}]})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        if payload.get("stream"):
            events = (
                'data: {"choices":[{"delta":{"content":"hola "}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"local"}}]}\n\n'
                "data: [DONE]\n\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(events)))
            self.end_headers()
            self.wfile.write(events)
            return
        self._send_json(
            {
                "model": LOCAL_MODEL,
                "choices": [
                    {"message": {"role": "assistant", "content": "hola local"}}
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 2,
                    "total_tokens": 4,
                },
            }
        )

    def _send_json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _fake_openai_server():
    _OpenAICompatibleHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_provider_against_real_openai_compatible_http_server() -> None:
    with _fake_openai_server() as base_url:
        settings = Settings(
            _env_file=None,
            local_llm_base_url=base_url,
            local_llm_model=LOCAL_MODEL,
        )
        provider = NemotronOmniLocalProvider(settings)
        request = ModelRequest(
            task_type="assistant",
            prompt="hola",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "estado", "parameters": {"type": "object"}},
                }
            ],
            response_format={"type": "json_object"},
        )

        assert provider.status_check().healthy is True
        response = provider.infer(
            request,
            model_name=LOCAL_MODEL,
            temperature=0.1,
            timeout_seconds=2,
        )
        chunks = list(
            provider.stream_infer(
                request,
                model_name=LOCAL_MODEL,
                temperature=0.1,
                timeout_seconds=2,
            )
        )

    assert response.content == "hola local"
    assert response.usage.total_tokens == 4
    assert "".join(chunk.text for chunk in chunks) == "hola local"
    assert chunks[-1].done is True
    assert len(_OpenAICompatibleHandler.requests) == 2
    assert all(
        item["model"] == LOCAL_MODEL for item in _OpenAICompatibleHandler.requests
    )
    assert all(
        item["tools"][0]["function"]["name"] == "estado"
        for item in _OpenAICompatibleHandler.requests
    )
    assert all(
        item["response_format"] == {"type": "json_object"}
        for item in _OpenAICompatibleHandler.requests
    )


def test_local_provider_serializes_tool_messages_without_internal_metadata() -> None:
    with _fake_openai_server() as base_url:
        settings = Settings(
            _env_file=None,
            local_llm_base_url=base_url,
            local_llm_model=LOCAL_MODEL,
        )
        provider = NemotronOmniLocalProvider(settings)
        request = ModelRequest(
            task_type="assistant",
            messages=[
                ModelMessage(role="user", content="calcula 781 x 94"),
                ModelMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ModelToolCall(
                            id="call-calculator-1",
                            name="calculator.evaluate",
                            arguments={"expression": "781 * 94"},
                        )
                    ],
                    metadata={"private_context": "must-not-cross-provider-boundary"},
                ),
                ModelMessage(
                    role="tool",
                    content='{"value": 73414}',
                    name="calculator.evaluate",
                    tool_call_id="call-calculator-1",
                ),
            ],
        )

        provider.infer(
            request,
            model_name=LOCAL_MODEL,
            temperature=0.1,
            timeout_seconds=2,
        )

    payload = _OpenAICompatibleHandler.requests[-1]
    assert payload["messages"] == [
        {"role": "user", "content": "calcula 781 x 94"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-calculator-1",
                    "type": "function",
                    "function": {
                        "name": "calculator.evaluate",
                        "arguments": '{"expression":"781 * 94"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"value": 73414}',
            "name": "calculator.evaluate",
            "tool_call_id": "call-calculator-1",
        },
    ]
    assert "metadata" not in json.dumps(payload)
    assert "must-not-cross-provider-boundary" not in json.dumps(payload)
