from __future__ import annotations

from jarvis.action_executor import ActionExecutor
from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController, AgentRunRequest
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


def _response(
    *, content: str = "", tool_calls: list[ModelToolCall] | None = None
) -> ModelResponse:
    return ModelResponse(
        provider_name="nemotron-test",
        provider_kind=ProviderKind.LOCAL,
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
