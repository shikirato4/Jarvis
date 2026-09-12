from __future__ import annotations

import pytest

from jarvis.action_executor import ActionExecutor
from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController, AgentRunRequest, AgentRunStatus
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.models_runtime.base import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
)
from jarvis.modules.web_module import build_web_search_action
from jarvis.modules.calculator_module import build_calculator_action
from jarvis.web_search import WebSearchHit, WebSearchResponse


class _Memory:
    def record_activity(self, **_payload) -> None:
        return


class _Web:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        self.queries.append(query)
        return WebSearchResponse(
            status="ok",
            provider="test-web",
            query=query,
            hits=[
                WebSearchHit(
                    title="Fuente actual",
                    url="https://example.test/current",
                    snippet="Resultado verificable.",
                    source="example.test",
                    rank=1,
                )
            ],
        )


class _UnavailableWeb:
    def __init__(self, status: str, error_code: str) -> None:
        self.status = status
        self.error_code = error_code

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        return WebSearchResponse(
            status=self.status,
            provider="brave",
            query=query,
            error_code=self.error_code,
            message=f"Brave Search: {self.error_code}",
        )


def _response(
    *,
    content: str = "",
    tool_calls: list[ModelToolCall] | None = None,
    provider_kind: ProviderKind = ProviderKind.LOCAL,
) -> ModelResponse:
    return ModelResponse(
        provider_name="nemotron-test",
        provider_kind=provider_kind,
        logical_model="general_assistant",
        model_name="nemotron-test",
        content=content,
        tool_calls=tool_calls or [],
        latency_ms=1.0,
    )


def test_web_search_is_a_nemotron_tool_not_a_pre_model_bypass() -> None:
    settings = Settings(_env_file=None)
    events = EventBus()
    web = _Web()
    registry = ActionRegistry()
    registry.register(build_web_search_action(web))
    router = ActionRouter(
        registry,
        settings,
        _Memory(),
        object(),
        events,
    )
    executor = ActionExecutor(router, event_sink=events.publish)
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="web-call-1",
                        name="web.search",
                        arguments={"query": "novedades Nemotron", "max_results": 3},
                    )
                ]
            ),
            _response(
                content=(
                    "Novedad verificada. Fuente: "
                    "https://example.test/current"
                )
            ),
        ]
    )

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="busca novedades Nemotron", max_steps=3))

    assert result.final_answer and "example.test/current" in result.final_answer
    assert web.queries == ["novedades Nemotron"]
    assert len(requests) == 2
    assert requests[0].messages[-1].role == "user"
    assert requests[1].messages[-1].role == "tool"
    assert "Fuente actual" in requests[1].messages[-1].content


@pytest.mark.parametrize("provider_kind", [ProviderKind.LOCAL, ProviderKind.REMOTE])
def test_shared_web_search_tool_reaches_local_and_online_models(provider_kind) -> None:
    settings = Settings(_env_file=None)
    events = EventBus()
    web = _Web()
    registry = ActionRegistry()
    registry.register(build_web_search_action(web))
    registry.register(build_calculator_action())
    executor = ActionExecutor(
        ActionRouter(registry, settings, _Memory(), object(), events),
        event_sink=events.publish,
    )
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _response(
                provider_kind=provider_kind,
                tool_calls=[
                    ModelToolCall(
                        id="shared-web-call",
                        name="web.search",
                        arguments={"query": "quién es actualmente el CEO de NVIDIA"},
                    )
                ],
            ),
            _response(content="Jensen Huang. https://example.test/current", provider_kind=provider_kind),
        ]
    )

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    result = AgentController(infer_model=infer, action_executor=executor).run(
        AgentRunRequest(user_message="Busca quién es actualmente el CEO de NVIDIA", max_steps=3)
    )

    first_tool_names = {
        schema["function"]["name"] for schema in requests[0].tools
    }
    assert result.final_answer and "Jensen Huang" in result.final_answer
    assert web.queries == ["quién es actualmente el CEO de NVIDIA"]
    assert {"web.search", "calculator.evaluate"} <= first_tool_names
    assert requests[1].messages[-1].role == "tool"


@pytest.mark.parametrize(
    ("web_status", "error_code"),
    [("unavailable", "missing_key"), ("timeout", "timeout")],
)
def test_online_model_continues_when_brave_is_unconfigured_or_temporarily_fails(
    web_status, error_code
) -> None:
    settings = Settings(_env_file=None)
    events = EventBus()
    registry = ActionRegistry()
    registry.register(build_web_search_action(_UnavailableWeb(web_status, error_code)))
    executor = ActionExecutor(
        ActionRouter(registry, settings, _Memory(), object(), events),
        event_sink=events.publish,
    )
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _response(
                provider_kind=ProviderKind.REMOTE,
                tool_calls=[
                    ModelToolCall(
                        id="degraded-web-call",
                        name="web.search",
                        arguments={"query": "noticias recientes de NVIDIA"},
                    )
                ],
            ),
            _response(
                content="No pude consultar la web, pero sigo disponible.",
                provider_kind=ProviderKind.REMOTE,
            ),
        ]
    )

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    result = AgentController(infer_model=infer, action_executor=executor).run(
        AgentRunRequest(user_message="Busca noticias recientes de NVIDIA", max_steps=3)
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "No pude consultar la web, pero sigo disponible."
    assert result.observations[0].status.value == "success"
    assert result.observations[0].data["error_code"] == error_code
    assert requests[1].messages[-1].role == "tool"


def test_desktop_request_construction_never_executes_web_search(monkeypatch) -> None:
    class _Panels:
        def compose(self):
            return {}

    def forbidden_provider():
        raise AssertionError("web provider ran before Nemotron")

    monkeypatch.setattr(
        "jarvis.desktop_runtime.chat.build_web_search_provider",
        forbidden_provider,
    )
    engine = DesktopChatEngine(
        object(),
        _Panels(),
        object(),
        Settings(_env_file=None),
    )

    request, web_result = engine._build_model_request(  # noqa: SLF001
        "busca novedades actuales de Nemotron"
    )

    assert web_result is None
    assert request.messages[-1].content == "busca novedades actuales de Nemotron"
    assert request.metadata["web_used"] is False


def test_desktop_preserves_web_sources_if_final_nemotron_call_fails() -> None:
    class _Panels:
        def compose(self):
            return {}

    settings = Settings(_env_file=None)
    web = _Web()
    registry = ActionRegistry()
    registry.register(build_web_search_action(web))
    executor = ActionExecutor(
        ActionRouter(registry, settings, _Memory(), object(), EventBus())
    )
    calls = 0

    def infer(_request: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response(
                tool_calls=[
                    ModelToolCall(
                        id="web-failure-1",
                        name="web.search",
                        arguments={"query": "novedades Nemotron"},
                    )
                ]
            )
        raise TimeoutError("private provider detail")

    controller = AgentController(infer_model=infer, action_executor=executor)
    engine = DesktopChatEngine(
        object(),
        _Panels(),
        object(),
        settings,
        agent_controller=controller,
    )

    response = engine.handle("busca novedades Nemotron")

    assert response.raw_result["status"] == "failed"
    assert "https://example.test/current" in response.message.content
    assert "private provider detail" not in response.message.content
