from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.action_executor import ActionExecutor, ActionRequest, ActionStatus
from jarvis.actions.discovery import DISCOVERY_ACTION, register_discovery_action
from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionDefinition, ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController, AgentRunRequest, AgentRunStatus
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ModelToolCall, ProviderKind


class _ArchivePayload(BaseModel):
    path: str = Field(min_length=1)


def _definition(name: str = "archive.bundle.inspect") -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description="Inspect a local compressed bundle and report its entries.",
        payload_model=_ArchivePayload,
        handler=lambda _context, payload: ActionResult(
            message="Archive inspected.", data={"path": payload.path, "entries": ["one.txt"]}
        ),
    )


def _runtime() -> tuple[ActionRegistry, ActionExecutor]:
    registry = ActionRegistry()
    register_discovery_action(registry)
    router = ActionRouter(
        registry, Settings(_env_file=None), object(), object(), EventBus()
    )
    return registry, ActionExecutor(router)


def _response(content: str = "", call: ModelToolCall | None = None) -> ModelResponse:
    return ModelResponse(
        provider_name="generic-test", provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant", model_name="generic-test",
        content=content, tool_calls=[call] if call else [], latency_ms=1,
    )


def test_discovery_is_read_only_and_uses_current_executable_registry() -> None:
    registry, executor = _runtime()
    initial = executor.execute(ActionRequest(
        run_id="r", call_id="before", action=DISCOVERY_ACTION,
        arguments={"query": "archive bundle"},
    ))
    assert initial.status is ActionStatus.SUCCESS
    assert initial.data["candidates"] == []
    registry.register(_definition())
    found = executor.execute(ActionRequest(
        run_id="r", call_id="after", action=DISCOVERY_ACTION,
        arguments={"query": "archive bundle"},
    ))
    assert [item["name"] for item in found.data["candidates"]] == ["archive.bundle.inspect"]
    assert "entries" not in found.data
    assert registry.unregister("archive.bundle.inspect")
    removed = executor.execute(ActionRequest(
        run_id="r", call_id="removed", action=DISCOVERY_ACTION,
        arguments={"query": "archive bundle"},
    ))
    assert removed.data["candidates"] == []
    assert "archive.bundle.inspect" not in {
        item["function"]["name"] for item in executor.tool_schemas()
    }


def test_unseen_capability_disclosed_on_next_agent_step_and_executes_normally() -> None:
    registry, executor = _runtime()
    registry.register(_definition())
    requests: list[ModelRequest] = []
    responses = iter([
        _response(call=ModelToolCall(
            id="discover", name=DISCOVERY_ACTION,
            arguments={"query": "archive bundle", "family": "archive"},
        )),
        _response(call=ModelToolCall(
            id="inspect", name="archive.bundle.inspect", arguments={"path": "fixture.zip"},
        )),
        _response(content="El archivo fixture.zip contiene one.txt."),
    ])

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    controller = AgentController(infer_model=infer, action_executor=executor)
    result = controller.run(AgentRunRequest(
        user_message="Examina este paquete comprimido: fixture.zip", max_steps=4,
    ))
    assert result.status is AgentRunStatus.COMPLETED
    assert [item.action for item in result.observations] == [
        DISCOVERY_ACTION, "archive.bundle.inspect",
    ]
    assert "one.txt" in result.final_answer
    assert DISCOVERY_ACTION in {item["function"]["name"] for item in requests[0].tools}
    assert "archive.bundle.inspect" in {
        item["function"]["name"] for item in requests[1].tools
    }


def test_removed_discovery_result_is_not_disclosed_on_next_request() -> None:
    registry, executor = _runtime()
    registry.register(_definition())
    controller = AgentController(infer_model=lambda _request: _response(), action_executor=executor)
    discovery = executor.execute(ActionRequest(
        run_id="r", call_id="d", action=DISCOVERY_ACTION,
        arguments={"query": "archive bundle"},
    ))
    registry.unregister("archive.bundle.inspect")
    from jarvis.models_runtime.base import ModelMessage

    prepared = controller.prepare_model_request(
        ModelRequest(task_type="agent", messages=[ModelMessage(role="user", content="Examine fixture.zip")]),
        discovered_names={item["name"] for item in discovery.data["candidates"]},
    )
    assert "archive.bundle.inspect" not in {
        item["function"]["name"] for item in prepared.tools
    }
