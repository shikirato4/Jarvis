from __future__ import annotations

from dataclasses import dataclass

import json

from jarvis.api.app import (
    ModelInferenceRequest,
    _action_catalog,
    _stream_model_events,
    _tool_catalog,
)
from jarvis.core.errors import ModelProviderError
from jarvis.models_runtime.base import StreamChunk


@dataclass
class _Definition:
    name: str
    description: str
    tags: tuple[str, ...]


class _Registry:
    def __init__(self, definitions: list[_Definition]) -> None:
        self._definitions = definitions
        self.calls = 0

    def list_tools(self) -> list[_Definition]:
        self.calls += 1
        return self._definitions

    def list_actions(self) -> list[_Definition]:
        self.calls += 1
        return self._definitions


class _JarvisCatalogStub:
    def __init__(self) -> None:
        self.tool_registry = _Registry([_Definition("calculator", "Calculate safely", ("math",))])
        self.action_registry = _Registry([_Definition("open_app", "Open an app", ("windows",))])

    def describe(self) -> dict[str, object]:
        raise AssertionError("static catalog must not initialize runtime health providers")


def test_tool_catalog_is_static_and_does_not_call_application_describe() -> None:
    jarvis = _JarvisCatalogStub()

    assert _tool_catalog(jarvis) == [
        {"name": "calculator", "description": "Calculate safely", "tags": ["math"]}
    ]
    assert jarvis.tool_registry.calls == 1


def test_action_catalog_is_static_and_does_not_call_application_describe() -> None:
    jarvis = _JarvisCatalogStub()

    assert _action_catalog(jarvis) == [
        {"name": "open_app", "description": "Open an app", "tags": ["windows"]}
    ]
    assert jarvis.action_registry.calls == 1


def test_stream_events_emit_real_chunks_and_terminal_state() -> None:
    class _Runtime:
        def stream_model(self, request):
            assert request.stream is False
            yield StreamChunk(text="Hola ", reasoning="Planifico el saludo.")
            yield StreamChunk(text="mundo", done=True, metadata={"chunks": 2})

    jarvis = type("Jarvis", (), {"runtime_service": _Runtime()})()
    body = ModelInferenceRequest(prompt="saluda", task_type="assistant")

    events = list(_stream_model_events(jarvis, body))
    payloads = [json.loads(item.removeprefix("data: ").strip()) for item in events]

    assert payloads == [
        {"text": "Hola ", "reasoning": "Planifico el saludo.", "done": False, "error": None, "metadata": {}},
        {"text": "mundo", "reasoning": "", "done": True, "error": None, "metadata": {"chunks": 2}},
    ]


def test_stream_events_return_domain_error_without_exception_details() -> None:
    class _Runtime:
        def stream_model(self, request):
            raise ModelProviderError("private provider diagnostic")
            yield  # pragma: no cover

    jarvis = type("Jarvis", (), {"runtime_service": _Runtime()})()

    [event] = list(
        _stream_model_events(
            jarvis, ModelInferenceRequest(prompt="hola", task_type="assistant")
        )
    )
    payload = json.loads(event.removeprefix("data: ").strip())

    assert payload["error"] == "model_provider_error"
    assert payload["done"] is True
    assert "private provider diagnostic" not in event
