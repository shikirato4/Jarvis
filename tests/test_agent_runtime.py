from __future__ import annotations

from pydantic import BaseModel

from jarvis.action_executor import (
    ActionExecutor,
    ActionObservation,
    ActionRequest,
    ActionStatus,
    TypedActionSecurityGate,
)
from jarvis.actions.models import ActionResult, ActionRisk, ExecutionStatus
from jarvis.actions.registry import ActionDefinition, ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import (
    AgentController,
    AgentOutcomeCode,
    AgentRunRequest,
    AgentRunStatus,
)
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.errors import JarvisError
from jarvis.desktop_runtime.intent_router import DesktopIntentDecision, DesktopIntentRouter
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.models_runtime.base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ProviderKind,
    StreamChunk,
)
from jarvis.models_runtime.output_completion import OutputCompletionPolicy
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
    *,
    content: str = "",
    tool_calls: list[ModelToolCall] | None = None,
    metadata: dict | None = None,
) -> ModelResponse:
    return ModelResponse(
        provider_name="nemotron-test",
        provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant",
        model_name="nemotron-test",
        content=content,
        tool_calls=tool_calls or [],
        latency_ms=1.0,
        metadata=metadata or {},
    )


class _ShadowObserver:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def observe(self, **payload):
        self.calls.append(payload)
        return list(payload["selected_tools"])


def test_agent_reports_real_final_tool_selection_to_shadow_router() -> None:
    class Payload(BaseModel):
        value: int

    executor, events = _executor(
        [
            ActionDefinition(
                name="calculator.evaluate",
                description="calculate",
                payload_model=Payload,
                handler=lambda _context, payload: ActionResult(
                    message="done", data={"value": payload.value}
                ),
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="calc-shadow-1",
                        name="calculator.evaluate",
                        arguments={"value": 73414},
                    )
                ]
            ),
            _response(content="73414"),
        ]
    )
    shadow = _ShadowObserver()
    controller = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
        event_sink=events.publish,
        shadow_observer=shadow,
    )

    result = controller.run(AgentRunRequest(user_message="calcula 781 por 94"))

    assert result.status is AgentRunStatus.COMPLETED
    assert len(shadow.calls) == 1
    assert shadow.calls[0]["selected_tools"] == ["calculator.evaluate"]
    assert shadow.calls[0]["success"] is True
    assert shadow.calls[0]["step_count"] == 1
    neural_events = [
        event
        for event in reversed(events.recent_events())
        if event["event_name"].startswith("neural_router.")
    ]
    assert [event["event_name"] for event in neural_events] == [
        "neural_router.started",
        "neural_router.completed",
    ]
    assert "calcula 781 por 94" not in repr(neural_events)


def test_agent_streams_model_tool_loop_and_final_answer() -> None:
    class Payload(BaseModel):
        expression: str

    executor, events = _executor(
        [
            ActionDefinition(
                name="calculator.evaluate",
                description="calculate",
                payload_model=Payload,
                handler=lambda _context, _payload: ActionResult(
                    message="done", data={"value": 73414}
                ),
            )
        ]
    )
    requests = []

    def stream(request, *, cancel_check=None):
        requests.append(request)
        assert cancel_check is not None
        if request.messages[-1].role == "tool":
            yield StreamChunk(
                text="El resultado ",
                metadata={"provider": "nemotron-test", "model": "nemotron-test"},
            )
            yield StreamChunk(
                text="es 73414.",
                metadata={"provider": "nemotron-test", "model": "nemotron-test"},
            )
            yield StreamChunk(
                done=True,
                metadata={
                    "finish_reason": "stop",
                    "stream_closed_normally": True,
                },
            )
            return
        yield StreamChunk(
            tool_calls=[
                ModelToolCall(
                    id="calculator-stream-1",
                    name="calculator.evaluate",
                    arguments={"expression": "781 * 94"},
                )
            ],
            metadata={"provider": "nemotron-test", "model": "nemotron-test"},
        )
        yield StreamChunk(
            done=True,
            metadata={
                "finish_reason": "tool_calls",
                "stream_closed_normally": True,
            },
        )

    streamed: list[str] = []
    controller = AgentController(
        infer_model=lambda _request: (_ for _ in ()).throw(
            AssertionError("streaming agent must not use synchronous inference")
        ),
        stream_model=stream,
        action_executor=executor,
        event_sink=events.publish,
    )

    result = controller.run(
        AgentRunRequest(
            user_message="calcula 781 por 94",
            stream_callback=streamed.append,
            cancellation_check=lambda: False,
        )
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "El resultado es 73414."
    assert streamed == ["El resultado ", "es 73414."]
    assert len(requests) == 2
    assert requests[0].tools
    assert requests[1].messages[-2].tool_calls[0].name == "calculator.evaluate"
    assert requests[1].messages[-1].role == "tool"
    assert '"value":73414' in requests[1].messages[-1].content


def test_agent_classifies_length_finish_as_truncated() -> None:
    executor, _events = _executor([])
    response = _response(content="respuesta parcial")
    response.metadata = {
        "completion_contract_enforced": True,
        "finish_reason": "length",
        "stream_closed_normally": True,
        "incomplete": True,
        "prompt_tokens": 4043,
        "completion_tokens": 53,
        "requested_max_tokens": 500,
    }

    result = AgentController(
        infer_model=lambda _request: response,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="continúa"))

    assert result.status is AgentRunStatus.TRUNCATED
    assert result.outcome_code is AgentOutcomeCode.MODEL_OUTPUT_TRUNCATED
    assert result.error_type == "OutputTokenLimitReached"
    assert result.final_answer == "respuesta parcial"
    assert result.finish_reason == "length"
    assert result.completion_tokens == 53


def test_agent_rejects_missing_finish_reason() -> None:
    executor, _events = _executor([])
    response = _response(content="sin terminal")
    response.metadata = {
        "completion_contract_enforced": True,
        "finish_reason": None,
        "stream_closed_normally": True,
        "incomplete": True,
    }

    result = AgentController(
        infer_model=lambda _request: response,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="hola"))

    assert result.status is AgentRunStatus.INTERRUPTED
    assert result.error_type == "MissingFinishReason"


def test_file_observation_compaction_preserves_facts_and_marks_truncation() -> None:
    compact = AgentController._compact_file_observation(  # noqa: SLF001
        {
            "operation": "files.read_file",
            "path": "C:/fixture/report.txt",
            "content": "x" * 2_500,
            "items": [{"name": str(index)} for index in range(25)],
            "metadata": {
                "name": "report.txt",
                "path": "C:/fixture/report.txt",
                "exists": True,
                "size": 2_500,
                "unrelated": "omit",
            },
            "unrelated": "omit",
        }
    )

    assert compact["path"] == "C:/fixture/report.txt"
    assert compact["metadata"] == {
        "name": "report.txt",
        "path": "C:/fixture/report.txt",
        "exists": True,
        "size": 2_500,
    }
    assert len(compact["content"]) == 2_000
    assert compact["content_truncated_for_model"] is True
    assert compact["content_original_chars"] == 2_500
    assert len(compact["items"]) == 20
    assert compact["items"][0] == {"name": "0"}
    assert compact["items_truncated_for_model"] is True
    assert compact["item_count"] == 25
    assert "unrelated" not in compact


def test_agent_stream_eof_never_masquerades_as_completed() -> None:
    executor, _events = _executor([])

    def stream(_request, *, cancel_check=None):
        yield StreamChunk(text="prefijo íntegro")
        # Generator exits without a terminal chunk.

    result = AgentController(
        infer_model=lambda _request: _response(content="unused"),
        stream_model=stream,
        action_executor=executor,
    ).run(
        AgentRunRequest(
            user_message="hola",
            stream_callback=lambda _delta: None,
        )
    )

    assert result.status is AgentRunStatus.INTERRUPTED
    assert result.error_type == "AbnormalStreamTermination"
    assert result.final_answer == "prefijo íntegro"
    assert result.stream_closed_normally is False


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
        "agent_controller.tool_calls_pending",
        "agent_controller.tool_requested",
        "agent_controller.tool_observed",
        "agent_controller.continuation_started",
        "agent_controller.model_requested",
        "agent_controller.tool_calls_pending",
        "agent_controller.tool_requested",
        "agent_controller.tool_observed",
        "agent_controller.continuation_started",
        "agent_controller.model_requested",
        "agent_controller.completed",
    ]


def test_agent_controller_retries_then_fails_when_required_tool_is_not_selected() -> None:
    executor, events = _executor([])
    requests: list[ModelRequest] = []

    def infer(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return _response(content="Puedes ejecutar el comando manualmente.")

    base_request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="ejecuta la acción")],
        metadata={"agent_tool_required": True},
    )
    result = AgentController(
        infer_model=infer,
        action_executor=executor,
        event_sink=events.publish,
    ).run(
        AgentRunRequest(
            user_message="ejecuta la acción",
            model_request=base_request,
        )
    )

    assert len(requests) == 2
    assert requests[1].messages[-1].role == "system"
    assert result.status is AgentRunStatus.FAILED
    assert result.outcome_code is AgentOutcomeCode.INVALID_MODEL_OUTPUT
    assert result.error_type == "MissingRequiredToolCall"
    assert result.final_answer == "Puedes ejecutar el comando manualmente."
    event_names = {
        event["event_name"] for event in events.recent_events()
    }
    assert "agent_controller.tool_selection_retry" in event_names


def test_truncated_structured_tool_call_retries_without_executing_partial_call() -> None:
    executions = 0

    def handler(_context, payload: _ValuePayload) -> ActionResult:
        nonlocal executions
        executions += 1
        return ActionResult(message="unexpected", data={"value": payload.value})

    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.value",
                description="Value tool.",
                payload_model=_ValuePayload,
                handler=handler,
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="partial-one", name="test.value", arguments={"value": 1}
                    )
                ],
                metadata={
                    "completion_contract_enforced": True,
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "incomplete": True,
                },
            ),
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="partial-two", name="test.value", arguments={"value": 2}
                    )
                ],
                metadata={
                    "completion_contract_enforced": True,
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "incomplete": True,
                },
            ),
        ]
    )
    base_request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content="ejecuta")],
        metadata={"agent_tool_required": True},
    )

    result = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    ).run(
        AgentRunRequest(
            user_message="ejecuta",
            model_request=base_request,
        )
    )

    assert executions == 0
    assert result.status is AgentRunStatus.FAILED
    assert result.outcome_code is AgentOutcomeCode.INVALID_MODEL_OUTPUT
    assert result.error_type == "TruncatedStructuredToolCall"
    assert result.model_call_count == 2


def test_agent_truncated_final_auto_continues_without_rerunning_tool() -> None:
    executions = 0

    def handler(_context, payload: _ValuePayload) -> ActionResult:
        nonlocal executions
        executions += 1
        return ActionResult(message="ok", data={"value": payload.value})

    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.value",
                description="Value tool.",
                payload_model=_ValuePayload,
                handler=handler,
            )
        ]
    )
    raw_responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="tool-once", name="test.value", arguments={"value": 7}
                    )
                ],
                metadata={
                    "completion_contract_enforced": True,
                    "finish_reason": "tool_calls",
                    "stream_closed_normally": True,
                    "incomplete": False,
                },
            ),
            _response(
                content="Resultado parcial ",
                metadata={
                    "completion_contract_enforced": True,
                    "finish_reason": "length",
                    "stream_closed_normally": True,
                    "incomplete": True,
                    "completion_tokens": 100,
                },
            ),
            _response(
                content="continuado.",
                metadata={
                    "completion_contract_enforced": True,
                    "finish_reason": "stop",
                    "stream_closed_normally": True,
                    "incomplete": False,
                    "completion_tokens": 40,
                },
            ),
        ]
    )
    policy = OutputCompletionPolicy(Settings(_env_file=None))

    def infer(request: ModelRequest) -> ModelResponse:
        prepared = policy.prepare_request(request)
        return policy.complete_response(
            prepared,
            next(raw_responses),
            infer_next=lambda _continuation: next(raw_responses),
        )

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="usa la herramienta"))

    assert executions == 1
    assert result.status is AgentRunStatus.COMPLETED
    assert result.final_answer == "Resultado parcial continuado."
    assert result.auto_continuation_count == 1
    assert result.model_request_count == 3
    assert result.segment_finish_reasons == ["tool_calls", "length", "stop"]
    assert result.segment_completion_tokens == [0, 100, 40]


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


def test_empty_final_after_successful_tool_is_continuation_failure() -> None:
    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.observe",
                description="Return one observation.",
                payload_model=_ValuePayload,
                handler=lambda _context, payload: ActionResult(
                    message="observed", data={"value": payload.value}
                ),
            )
        ]
    )
    final = _response(content="")
    final.metadata = {
        "completion_contract_enforced": True,
        "finish_reason": "stop",
        "stream_closed_normally": True,
        "incomplete": False,
    }
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="observe-1",
                        name="test.observe",
                        arguments={"value": 7},
                    )
                ]
            ),
            final,
        ]
    )

    result = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    ).run(AgentRunRequest(user_message="observa"))

    assert result.status is AgentRunStatus.FAILED
    assert result.outcome_code is AgentOutcomeCode.MODEL_CONTINUATION_FAILURE
    assert result.error_type == "EmptyContinuationFinal"
    assert result.continuation_requested is True
    assert result.assistant_final_present is False
    text = DesktopChatEngine._agent_result_text(  # noqa: SLF001
        DesktopChatEngine.__new__(DesktopChatEngine), result
    )
    assert "observación" in text
    assert "salida valida" not in text


def test_failed_tool_cannot_be_classified_as_success_even_if_model_claims_done() -> None:
    executor, _events = _executor(
        [
            ActionDefinition(
                name="test.fail",
                description="Fail deliberately.",
                payload_model=_ValuePayload,
                handler=lambda _context, _payload: ActionResult(
                    status=ExecutionStatus.FAILED,
                    message="real tool failure",
                ),
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="fail-1",
                        name="test.fail",
                        arguments={"value": 1},
                    )
                ]
            ),
            _response(content="Hecho."),
        ]
    )

    result = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    ).run(AgentRunRequest(user_message="hazlo"))

    assert result.status is AgentRunStatus.FAILED
    assert result.outcome_code is AgentOutcomeCode.TOOL_EXECUTION_ERROR
    assert result.observations[-1].status is ActionStatus.FAILED
    visible = DesktopChatEngine._agent_result_text(  # noqa: SLF001
        DesktopChatEngine.__new__(DesktopChatEngine), result
    )
    assert "herramienta no pudo completar" in visible


def test_security_gate_denial_remains_a_structured_agent_outcome() -> None:
    class _TerminalPayload(BaseModel):
        command: str

    executor, _events = _executor(
        [
            ActionDefinition(
                name="operations.run_command",
                description="Run a terminal command.",
                payload_model=_TerminalPayload,
                handler=lambda _context, _payload: ActionResult(message="should not run"),
                risk=ActionRisk.SENSITIVE,
            )
        ]
    )
    responses = iter(
        [
            _response(
                tool_calls=[
                    ModelToolCall(
                        id="blocked-1",
                        name="operations.run_command",
                        arguments={"command": "Remove-Item C:/Windows -Recurse -Force"},
                    )
                ]
            ),
            _response(content="No se ejecutó."),
        ]
    )

    result = AgentController(
        infer_model=lambda _request: next(responses),
        action_executor=executor,
    ).run(AgentRunRequest(user_message="borra windows"))

    assert result.status is AgentRunStatus.BLOCKED
    assert result.outcome_code is AgentOutcomeCode.SECURITY_DENIED
    assert result.observations[-1].status is ActionStatus.BLOCKED


def test_file_final_omits_paths_not_present_in_real_observations() -> None:
    observations = [
        ActionObservation(
            run_id="files-run",
            call_id="files-call",
            action="files.search",
            status=ActionStatus.SUCCESS,
            message="searched",
            data={
                "path": "C:/Users/example/Documents",
                "items": [],
                "scanned": 30,
            },
        )
    ]
    response = (
        "No encontré coincidencias. Prueba en "
        "`C:\\Users\\Example\\AppData\\Local\\InventedGame`."
    )

    grounded = AgentController._ground_file_paths(response, observations)  # noqa: SLF001

    assert "InventedGame" not in grounded
    assert "ruta no verificada omitida" in grounded
    assert "no apareció en las observaciones" in grounded
