from __future__ import annotations

from pydantic import BaseModel

from jarvis.action_executor import (
    ActionExecutor,
    ActionRequest,
    ActionStatus,
    TypedActionSecurityGate,
)
from jarvis.actions.models import ActionResult, ActionRisk
from jarvis.actions.registry import ActionDefinition, ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController, AgentRunRequest, AgentRunStatus
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.errors import JarvisError
from jarvis.desktop_runtime.intent_router import DesktopIntentDecision, DesktopIntentRouter
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.models_runtime.base import (
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
)
from jarvis.modules.autonomy_module import AutonomyModule
from jarvis.modules.desktop_agent_module import DesktopAgentModule
from jarvis.modules.interface_module import InterfaceModule
from jarvis.modules.operations_module import OperationsModule
from jarvis.modules.system_module import SystemModule


class _ValuePayload(BaseModel):
    value: int


class _SecretPayload(BaseModel):
    password: str


class _Memory:
    def __init__(self) -> None:
        self.activities: list[dict[str, object]] = []

    def record_activity(self, **payload) -> None:
        self.activities.append(payload)


class _UnavailableAuditMemory:
    def record_activity(self, **_payload) -> None:
        raise OSError("audit database unavailable")


class _Models:
    pass


def _executor(
    definitions: list[ActionDefinition],
) -> tuple[ActionExecutor, EventBus]:
    registry = ActionRegistry()
    for definition in definitions:
        registry.register(definition)
    events = EventBus()
    router = ActionRouter(
        registry,
        Settings(_env_file=None),
        _Memory(),
        _Models(),
        events,
    )
    return (
        ActionExecutor(
            router,
            security_gate=TypedActionSecurityGate(),
            event_sink=events.publish,
        ),
        events,
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


def test_os_and_ui_mutations_are_never_registered_as_safe_actions() -> None:
    registry = ActionRegistry()
    SystemModule(object()).register_actions(registry)
    InterfaceModule(object(), object()).register_actions(registry)
    OperationsModule(object()).register_actions(registry)
    DesktopAgentModule(lambda: object()).register_actions(registry)
    AutonomyModule(object()).register_actions(registry)

    expected_sensitive = {
        "system.open_target",
        "system.open_path",
        "system.launch_application",
        "system.reveal_target",
        "interface.focus_window",
        "interface.write_text",
        "interface.insert_blocks",
        "interface.move_mouse",
        "interface.click_mouse",
        "interface.keyboard_shortcut",
        "interface.contextual_direct_write",
        "operations.run_command",
        "desktop_agent.run_mission",
        "autonomy.start_mission",
        "autonomy.step_mission",
        "autonomy.stop_mission",
    }

    assert {
        definition.name
        for definition in registry.list_actions()
        if definition.risk is ActionRisk.SENSITIVE
    } >= expected_sensitive
    assert registry.get("system.search_resources").risk is ActionRisk.SAFE  # type: ignore[union-attr]
    assert registry.get("system.resolve_target").risk is ActionRisk.SAFE  # type: ignore[union-attr]
    assert registry.get("interface.active_window").risk is ActionRisk.SAFE  # type: ignore[union-attr]
    assert registry.get("autonomy.plan_mission").risk is ActionRisk.SAFE  # type: ignore[union-attr]


def test_blocked_vision_capture_is_partial_never_completed() -> None:
    class _Receipt:
        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"active_window": {"title": "Protected window"}}

    class _Runtime:
        def vision_describe_active_window(self):
            raise JarvisError(
                "capture blocked for a sensitive window",
                component="vision_runtime",
                code="capture_failed",
                recoverable=True,
            )

        def ui_active_window(self):
            return _Receipt()

    bridge = type("Bridge", (), {"runtime": _Runtime()})()
    result, _summary, _route = DesktopIntentRouter(bridge, None).execute(
        DesktopIntentDecision(category="window_context", prompt="que ves")
    )

    assert result["status"] == "partial"
    assert result["status"] != "completed"
    assert result["degraded"] is True


def test_action_executor_validates_and_emits_private_success_events() -> None:
    calls: list[int] = []

    def handler(_context, payload: _ValuePayload) -> ActionResult:
        calls.append(payload.value)
        return ActionResult(message="done", data={"doubled": payload.value * 2})

    executor, events = _executor(
        [
            ActionDefinition(
                name="test.double",
                description="Double a value.",
                payload_model=_ValuePayload,
                handler=handler,
                risk=ActionRisk.SAFE,
                idempotent=True,
            )
        ]
    )
    request = ActionRequest(
        run_id="run-1",
        call_id="call-1",
        action="test.double",
        arguments={"value": 21},
        correlation_id="corr-1",
    )

    first = executor.execute(request)
    repeated = executor.execute(request)

    assert first.status is ActionStatus.SUCCESS
    assert first.data == {"doubled": 42}
    assert repeated.deduplicated is True
    assert calls == [21]
    safe_events = [
        event
        for event in reversed(events.recent_events())
        if event["event_name"].startswith("action_executor.")
    ]
    assert [event["event_name"] for event in safe_events] == [
        "action_executor.started",
        "action_executor.succeeded",
        "action_executor.deduplicated",
    ]
    rendered = repr([event["payload"] for event in safe_events])
    assert "21" not in rendered
    assert "doubled" not in rendered


def test_sensitive_action_requires_scoped_single_use_confirmation() -> None:
    calls: list[int] = []

    def handler(_context, payload: _ValuePayload) -> ActionResult:
        calls.append(payload.value)
        return ActionResult(message="changed")

    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.change",
                description="Change protected state.",
                payload_model=_ValuePayload,
                handler=handler,
                risk=ActionRisk.SENSITIVE,
                idempotent=False,
            )
        ]
    )
    request = ActionRequest(
        run_id="run-sensitive",
        call_id="call-sensitive",
        action="test.change",
        arguments={"value": 7},
    )

    pending = executor.execute(request)
    assert pending.status is ActionStatus.CONFIRMATION_REQUIRED
    assert pending.confirmation_id
    assert calls == []

    executor.approve_confirmation(pending.confirmation_id)
    approved = executor.execute(
        request.model_copy(update={"confirmation_token": pending.confirmation_id})
    )
    replay = executor.execute(
        request.model_copy(
            update={
                "call_id": "call-sensitive-replay",
                "confirmation_token": pending.confirmation_id,
            }
        )
    )

    assert approved.status is ActionStatus.SUCCESS
    assert replay.status is ActionStatus.CONFIRMATION_REQUIRED
    assert calls == [7]


def test_agent_can_synthesize_after_using_its_last_allowed_tool_step() -> None:
    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.once",
                description="Use one tool.",
                payload_model=_ValuePayload,
                handler=lambda _context, payload: ActionResult(
                    message="ok", data={"value": payload.value}
                ),
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="only-tool",
                        name="test.once",
                        arguments={"value": 1},
                    )
                ]
            ),
            _response(content="final after one tool"),
        ]
    )

    result = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    ).run(AgentRunRequest(user_message="one tool", max_steps=1))

    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "final after one tool"
    assert result.step_count == 1


def test_agent_confirmation_resumes_same_run_and_context() -> None:
    calls: list[int] = []

    def handler(_context, payload: _ValuePayload) -> ActionResult:
        calls.append(payload.value)
        return ActionResult(message="changed", data={"value": payload.value})

    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.confirmed_change",
                description="Change state.",
                payload_model=_ValuePayload,
                handler=handler,
                risk=ActionRisk.SENSITIVE,
                idempotent=False,
            )
        ]
    )
    requests: list[ModelRequest] = []

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        if len(requests) == 1:
            return _response(
                tool_calls=[
                    ModelToolCall(
                        id="confirmed-tool",
                        name="test.confirmed_change",
                        arguments={"value": 9},
                    )
                ]
            )
        return _response(content="change confirmed")

    controller = AgentController(infer_model=infer, action_executor=executor)
    waiting = controller.run(
        AgentRunRequest(user_message="change it", run_id="confirmation-run")
    )

    assert waiting.status is AgentRunStatus.WAITING_CONFIRMATION
    assert waiting.pending_confirmation is not None
    assert calls == []

    resumed = controller.resume_confirmation(
        waiting.run_id,
        waiting.pending_confirmation.confirmation_id or "",
    )

    assert resumed.status is AgentRunStatus.COMPLETED
    assert resumed.run_id == waiting.run_id
    assert resumed.final_answer == "change confirmed"
    assert resumed.step_count == 1
    assert calls == [9]
    assert requests[-1].messages[-1].role == "tool"
    assert "confirmation_required" not in requests[-1].messages[-1].content


def test_desktop_yes_resumes_pending_agent_confirmation() -> None:
    calls: list[int] = []
    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.desktop_change",
                description="Change desktop state.",
                payload_model=_ValuePayload,
                handler=lambda _context, payload: (
                    calls.append(payload.value)
                    or ActionResult(message="changed", data={"value": payload.value})
                ),
                risk=ActionRisk.SENSITIVE,
                idempotent=False,
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="desktop-confirm-call",
                        name="test.desktop_change",
                        arguments={"value": 4},
                    )
                ]
            ),
            _response(content="desktop change complete"),
        ]
    )
    controller = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    )
    panels = type("Panels", (), {"compose": lambda self: {}})()
    engine = DesktopChatEngine(
        object(),
        panels,
        object(),
        Settings(_env_file=None),
        agent_controller=controller,
    )

    waiting = engine.handle("change desktop state")
    resumed = engine.handle("sí")

    assert "confirmación" in waiting.message.content
    assert resumed.message.content == "desktop change complete"
    assert resumed.raw_result["status"] == "completed"
    assert calls == [4]


def test_action_audit_events_and_activity_never_store_secret_values() -> None:
    sentinel = "TOP-SECRET-ACTION-VALUE-81"
    registry = ActionRegistry()
    registry.register(
        ActionDefinition(
            name="test.secret",
            description="Handle a secret without persisting it.",
            payload_model=_SecretPayload,
            handler=lambda _context, _payload: ActionResult(
                message="checked",
                data={"echo": sentinel},
            ),
        )
    )
    events = EventBus()
    memory = _Memory()
    router = ActionRouter(
        registry,
        Settings(_env_file=None),
        memory,
        _Models(),
        events,
    )

    receipt = router.execute("test.secret", {"password": sentinel})

    assert receipt.status.value == "success"
    assert sentinel not in repr(events.recent_events())
    assert sentinel not in repr(memory.activities)
    assert memory.activities[0]["payload"]["field_names"] == ["password"]


def test_successful_action_is_not_failed_by_unavailable_audit_persistence() -> None:
    registry = ActionRegistry()
    registry.register(
        ActionDefinition(
            name="test.value",
            description="Return a value.",
            payload_model=_ValuePayload,
            handler=lambda _context, payload: ActionResult(
                message="completed", data={"value": payload.value}
            ),
        )
    )
    events = EventBus()
    router = ActionRouter(
        registry,
        Settings(_env_file=None),
        _UnavailableAuditMemory(),
        _Models(),
        events,
    )

    receipt = router.execute("test.value", {"value": 7})

    assert receipt.status.value == "success"
    assert receipt.data == {"value": 7}
    assert any(
        item["event_name"] == "action.audit_persistence_failed"
        for item in events.recent_events()
    )


def test_agent_controller_runs_two_tools_then_returns_nemotron_final() -> None:
    def tool_handler(_context, payload: _ValuePayload) -> ActionResult:
        return ActionResult(message="ok", data={"value": payload.value})

    executor, events = _executor(
        [
            ActionDefinition(
                name="test.first",
                description="First test tool.",
                payload_model=_ValuePayload,
                handler=tool_handler,
            ),
            ActionDefinition(
                name="test.second",
                description="Second test tool.",
                payload_model=_ValuePayload,
                handler=tool_handler,
            ),
        ]
    )
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="tool-1", name="test.first", arguments={"value": 1}
                    )
                ]
            ),
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="tool-2", name="test.second", arguments={"value": 2}
                    )
                ]
            ),
            _response(content="Resultado final de Nemotron."),
        ]
    )

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    controller = AgentController(
        infer_model=infer,
        action_executor=executor,
        event_sink=events.publish,
    )
    result = controller.run(
        AgentRunRequest(user_message="haz dos pasos", max_steps=4)
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "Resultado final de Nemotron."
    assert result.step_count == 2
    assert len(requests) == 3
    assert requests[0].tools
    assert requests[1].messages[-1].role == "tool"
    assert requests[1].messages[-1].tool_call_id == "tool-1"
    assert requests[2].messages[-1].tool_call_id == "tool-2"
    agent_events = [
        event["event_name"]
        for event in reversed(events.recent_events())
        if event["event_name"].startswith("agent_controller.")
    ]
    assert agent_events == [
        "agent_controller.started",
        "agent_controller.model_requested",
        "agent_controller.tool_requested",
        "agent_controller.tool_observed",
        "agent_controller.model_requested",
        "agent_controller.tool_requested",
        "agent_controller.tool_observed",
        "agent_controller.model_requested",
        "agent_controller.completed",
    ]


def test_agent_controller_replans_after_tool_failure() -> None:
    def failing(_context, _payload: _ValuePayload) -> ActionResult:
        raise RuntimeError("private failure detail")

    def alternative(_context, payload: _ValuePayload) -> ActionResult:
        return ActionResult(message="alternative ok", data={"value": payload.value})

    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.fail",
                description="Fail safely.",
                payload_model=_ValuePayload,
                handler=failing,
            ),
            ActionDefinition(
                name="test.alternative",
                description="Alternative.",
                payload_model=_ValuePayload,
                handler=alternative,
            ),
        ]
    )
    requests: list[ModelRequest] = []
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(id="fail-1", name="test.fail", arguments={"value": 1})
                ]
            ),
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="alternative-1",
                        name="test.alternative",
                        arguments={"value": 2},
                    )
                ]
            ),
            _response(content="Me recupere con una alternativa."),
        ]
    )

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return next(responses)

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="usa una alternativa si falla", max_steps=4))

    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "Me recupere con una alternativa."
    failed_observation = requests[1].messages[-1].content
    assert '"status":"failed"' in failed_observation
    assert "private failure detail" not in failed_observation


def test_agent_controller_cancellation_stops_before_model_call_and_emits_event() -> None:
    executor, events = _executor([])
    model_calls = 0

    def infer(_request: ModelRequest) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return _response(content="should not run")

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
        event_sink=events.publish,
    ).run(
        AgentRunRequest(
            user_message="cancel this",
            cancellation_check=lambda: True,
        )
    )

    assert result.status is AgentRunStatus.CANCELLED
    assert model_calls == 0
    names = [
        event["event_name"]
        for event in reversed(events.recent_events())
        if event["event_name"].startswith("agent_controller.")
    ]
    assert names == ["agent_controller.started", "agent_controller.cancelled"]


def test_agent_controller_deadline_is_explicit_and_emits_event() -> None:
    executor, events = _executor([])
    ticks = iter([0.0, 2.0])

    result = AgentController(
        infer_model=lambda _request: _response(content="should not run"),
        action_executor=executor,
        event_sink=events.publish,
        clock=lambda: next(ticks),
    ).run(AgentRunRequest(user_message="timed", timeout_seconds=1.0))

    assert result.status is AgentRunStatus.TIMED_OUT
    names = [
        event["event_name"]
        for event in reversed(events.recent_events())
        if event["event_name"].startswith("agent_controller.")
    ]
    assert names == ["agent_controller.started", "agent_controller.timed_out"]


def test_agent_controller_blocks_repeated_action_loop() -> None:
    def handler(_context, payload: _ValuePayload) -> ActionResult:
        return ActionResult(message="ok", data={"value": payload.value})

    executor, events = _executor(
        [
            ActionDefinition(
                name="test.repeat",
                description="Repeated tool.",
                payload_model=_ValuePayload,
                handler=handler,
            )
        ]
    )
    model_calls = 0

    def infer(_request: ModelRequest) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        return _response(
            tool_calls=[
                ModelToolCall(
                    id=f"repeat-{model_calls}",
                    name="test.repeat",
                    arguments={"value": 1},
                )
            ]
        )

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
        event_sink=events.publish,
        max_repeated_signature=2,
    ).run(AgentRunRequest(user_message="do not loop", max_steps=8))

    assert result.status is AgentRunStatus.BLOCKED
    assert result.error_type == "RepeatedActionLoop"
    assert result.step_count == 2
    assert model_calls == 3
    assert any(
        event["event_name"] == "agent_controller.blocked"
        for event in events.recent_events()
    )
