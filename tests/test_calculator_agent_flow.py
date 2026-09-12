from __future__ import annotations

from jarvis.action_executor import ActionExecutor, TypedActionSecurityGate
from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.models_runtime.base import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
)
from jarvis.modules.calculator_module import build_calculator_action


class _Memory:
    def record_activity(self, **_payload) -> None:
        return


class _Panels:
    @staticmethod
    def compose():
        return None


class _CaptureRuntime:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = iter(responses)
        self.requests: list[ModelRequest] = []
        self.science_calls = 0

    def infer_model(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return next(self.responses)

    def science_solve(self, _request):
        self.science_calls += 1
        raise AssertionError("normal calculator chat must not bypass Nemotron")


class _Bridge:
    def __init__(self, runtime: _CaptureRuntime) -> None:
        self.runtime = runtime


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


def test_calculation_flows_nemotron_tool_result_nemotron() -> None:
    settings = Settings(_env_file=None, voice_cleanup_enabled=False)
    events = EventBus(history_limit=100)
    registry = ActionRegistry()
    registry.register(build_calculator_action())
    router = ActionRouter(
        registry,
        settings,
        _Memory(),
        object(),
        events,
    )
    executor = ActionExecutor(
        router,
        security_gate=TypedActionSecurityGate(),
        event_sink=events.publish,
    )
    runtime = _CaptureRuntime(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="calculator-call-1",
                        name="calculator.evaluate",
                        arguments={"expression": "781 * 94"},
                    )
                ]
            ),
            _response(content="781 × 94 es 73414."),
        ]
    )
    controller = AgentController(
        infer_model=runtime.infer_model,
        action_executor=executor,
        event_sink=events.publish,
    )
    engine = DesktopChatEngine(
        _Bridge(runtime),
        _Panels(),
        object(),
        settings,
        agent_controller=controller,
    )

    response = engine.handle("calcula 781 × 94", correlation_id="calc-flow-1")

    assert response.message.content == "781 × 94 es 73414."
    assert runtime.science_calls == 0
    assert len(runtime.requests) == 2
    first, second = runtime.requests
    assert first.messages[-1].role == "user"
    assert first.messages[-1].content == "calcula 781 × 94"
    assert any(
        schema["function"]["name"] == "calculator.evaluate"
        for schema in first.tools
    )
    assert second.messages[-2].tool_calls[0].name == "calculator.evaluate"
    assert second.messages[-1].role == "tool"
    assert second.messages[-1].tool_call_id == "calculator-call-1"
    assert '"value":73414' in second.messages[-1].content
    names = [
        event["event_name"]
        for event in reversed(events.recent_events())
        if event["event_name"].startswith(
            ("agent_controller.", "action_executor.")
        )
    ]
    assert names == [
        "agent_controller.started",
        "agent_controller.model_requested",
        "agent_controller.tool_calls_pending",
        "agent_controller.tool_requested",
        "action_executor.started",
        "action_executor.succeeded",
        "agent_controller.tool_observed",
        "agent_controller.continuation_started",
        "agent_controller.model_requested",
        "agent_controller.completed",
    ]
