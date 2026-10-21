from __future__ import annotations

from pydantic import BaseModel

from jarvis.action_executor import (
    ActionExecutor,
    ActionRequest,
    ActionStatus,
    TypedActionSecurityGate,
)
from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionDefinition, ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy
from jarvis.config import Settings
from jarvis.core.capabilities import CapabilityRegistry
from jarvis.core.events import EventBus
from jarvis.modules.windows_module import WindowsModule
from jarvis.runtime_activity import RuntimeActivityStore
from jarvis.windows_runtime import (
    InMemoryWindowsControlBackend,
    WindowCandidateState,
    WindowRef,
    WindowTarget,
    WindowsErrorCode,
    WindowsRuntimeError,
    WindowsRuntimeService,
    classify_window_candidate,
)


class _Memory:
    def __init__(self) -> None:
        self.receipts = []

    def record_activity(self, **payload) -> None:
        self.receipts.append(payload)


class _Models:
    pass


def _runtime_stack():
    backend = InMemoryWindowsControlBackend()
    module = WindowsModule(WindowsRuntimeService(backend))
    actions = ActionRegistry()
    capabilities = CapabilityRegistry()
    module.register_actions(actions)
    module.register_capabilities(capabilities)
    events = EventBus()
    memory = _Memory()
    activity = RuntimeActivityStore(events)
    router = ActionRouter(
        actions,
        Settings(_env_file=None),
        memory,
        _Models(),
        events,
    )
    executor = ActionExecutor(
        router,
        security_gate=TypedActionSecurityGate(),
        event_sink=events.publish,
    )
    return backend, module, actions, capabilities, memory, activity, executor


def _execute(executor, action, arguments=None, *, call_id="call", token=None):
    return executor.execute(
        ActionRequest(
            run_id="windows-run",
            call_id=call_id,
            action=action,
            arguments=arguments or {},
            confirmation_token=token,
        )
    )


def test_environment_process_window_and_foreground_observations_are_bounded() -> None:
    backend = InMemoryWindowsControlBackend()

    environment = backend.inspect_environment()
    processes = backend.list_processes(limit=1, interactive_only=True)
    windows = backend.list_windows(limit=1)
    foreground = backend.foreground_window()

    assert environment.verified is True
    assert environment.environment.build == "26100"
    assert len(processes.processes) == 1 and processes.truncated is True
    assert len(windows.windows) == 1 and windows.count == 2
    assert foreground.target.title == "Notes"
    assert foreground.target.runtime_id == "win:10:0x1"


def test_window_ref_is_not_title_only_and_exact_runtime_identity_resolves() -> None:
    backend = InMemoryWindowsControlBackend()
    observed = backend.resolve_window(WindowTarget(runtime_id="win:20:0x2"))

    assert observed.pid == 20
    assert observed.handle == "0x2"
    assert observed.process_name == "notepad.exe"
    assert observed.class_name is None


def test_resolver_reports_missing_ambiguous_and_stale_targets() -> None:
    backend = InMemoryWindowsControlBackend()
    backend.windows.append(
        WindowRef(
            runtime_id="win:21:0x3",
            handle="0x3",
            pid=21,
            process_name="notepad.exe",
            application="Notepad",
            title="Second - Notepad",
        )
    )

    try:
        backend.resolve_window(WindowTarget(query="notepad"))
    except WindowsRuntimeError as exc:
        assert exc.code is WindowsErrorCode.AMBIGUOUS_TARGET
        assert len(exc.candidates) == 2
    else:
        raise AssertionError("ambiguous target was accepted")

    for target, expected in (
        (WindowTarget(query="missing"), WindowsErrorCode.TARGET_NOT_FOUND),
        (WindowTarget(runtime_id="win:99:0xff"), WindowsErrorCode.TARGET_STALE),
    ):
        try:
            backend.resolve_window(target)
        except WindowsRuntimeError as exc:
            assert exc.code is expected
        else:
            raise AssertionError(f"{expected} was not raised")


def test_focus_minimize_maximize_restore_are_verified_and_idempotent() -> None:
    backend = InMemoryWindowsControlBackend()
    target = WindowTarget(runtime_id="win:20:0x2")

    focused = backend.focus_window(target, timeout_seconds=1)
    minimized = backend.minimize_window(target, timeout_seconds=1)
    minimized_again = backend.minimize_window(target, timeout_seconds=1)
    maximized = backend.maximize_window(target, timeout_seconds=1)
    restored = backend.restore_window(target, timeout_seconds=1)

    assert focused.verified and focused.after.foreground
    assert minimized.after.minimized and minimized_again.after.minimized
    assert maximized.after.maximized and not maximized.after.minimized
    assert not restored.after.maximized and not restored.after.minimized
    assert all(
        any(item.runtime_id == target.runtime_id for item in backend.list_windows(limit=30).windows)
        for _state in (focused, minimized, minimized_again, maximized, restored)
    )


def test_window_classification_keeps_minimized_apps_and_suppresses_helpers() -> None:
    base = dict(
        exists=True,
        title="Untitled - Notepad",
        process_name="notepad.exe",
        class_name="Notepad",
        visible=True,
        minimized=False,
        maximized=False,
        owner_handle=0,
        tool_window=False,
        cloaked=False,
    )
    normal = classify_window_candidate(WindowCandidateState(**base))
    minimized = classify_window_candidate(
        WindowCandidateState(**(base | {"visible": False, "minimized": True}))
    )
    hidden = classify_window_candidate(
        WindowCandidateState(**(base | {"visible": False}))
    )
    helper = classify_window_candidate(
        WindowCandidateState(**(base | {"class_name": "WorkerW"}))
    )
    owned = classify_window_candidate(
        WindowCandidateState(**(base | {"owner_handle": 123}))
    )

    assert normal.targetable and normal.list_eligible
    assert minimized.targetable and minimized.list_eligible
    assert hidden.suppression_reasons == ("hidden_non_minimized",)
    assert "helper_class" in helper.suppression_reasons
    assert "owned_window" in owned.suppression_reasons
    assert not hidden.targetable and not helper.targetable and not owned.targetable


def test_hidden_helper_is_neither_listed_nor_resolved() -> None:
    backend = InMemoryWindowsControlBackend()
    helper = WindowRef(
        runtime_id="win:99:0x99",
        handle="0x99",
        pid=99,
        process_name="helper.exe",
        application="Notepad",
        title="Untitled - Notepad helper",
        visible=False,
    )
    backend.windows.append(helper)
    backend.suppressed_runtime_ids.add(helper.runtime_id)

    assert helper.runtime_id not in {
        item.runtime_id for item in backend.list_windows(limit=30).windows
    }
    try:
        backend.resolve_window(WindowTarget(runtime_id=helper.runtime_id))
    except WindowsRuntimeError as exc:
        assert exc.code is WindowsErrorCode.TARGET_STALE
    else:
        raise AssertionError("suppressed helper became actionable")


def test_recreated_hwnd_can_rebind_only_with_unique_semantic_identity() -> None:
    backend = InMemoryWindowsControlBackend()
    rebound = backend.resolve_window(
        WindowTarget(runtime_id="win:20:0xdead", process_name="notepad.exe")
    )
    assert rebound.runtime_id == "win:20:0x2"

    backend.windows.append(
        WindowRef(
            runtime_id="win:21:0x3",
            handle="0x3",
            pid=21,
            process_name="notepad.exe",
            application="Notepad",
            title="Second - Notepad",
        )
    )
    try:
        backend.resolve_window(
            WindowTarget(runtime_id="win:20:0xdead", process_name="notepad.exe")
        )
    except WindowsRuntimeError as exc:
        assert exc.code is WindowsErrorCode.AMBIGUOUS_TARGET
    else:
        raise AssertionError("recreated duplicate app was guessed")


def test_safe_catalog_launch_rejects_commands_and_verifies_new_window() -> None:
    backend = InMemoryWindowsControlBackend()

    observed = backend.launch_application("Notepad", timeout_seconds=1)
    assert observed.verified and observed.after.process_name == "notepad.exe"
    assert backend.launched == ["notepad.exe"]

    try:
        backend.launch_application("cmd /c whoami", timeout_seconds=1)
    except WindowsRuntimeError as exc:
        assert exc.code is WindowsErrorCode.OPERATION_NOT_SUPPORTED
    else:
        raise AssertionError("arbitrary command launch was accepted")


def test_actions_are_registered_with_risk_and_cancellation_metadata() -> None:
    _, _, actions, capabilities, _, _, _ = _runtime_stack()

    names = {item.name for item in actions.list_actions()}
    assert "windows.window.focus" in names
    assert "windows.application.launch" in names
    assert actions.get("windows.window.focus").idempotent is True
    assert actions.get("windows.window.focus").supports_cancellation is True
    assert "minimized OR maximized" in actions.get("windows.window.restore").description
    assert actions.get("windows.application.launch").idempotent is False
    assert actions.get("windows.application.launch").risk.value == "sensitive"
    assert capabilities.get("windows").descriptor.action_names == tuple(sorted(names)) or set(
        capabilities.get("windows").descriptor.action_names
    ) == names


def test_security_gate_executor_receipt_and_activity_graph_cover_window_action() -> None:
    _, _, _, _, memory, activity, executor = _runtime_stack()

    observation = _execute(
        executor,
        "windows.window.focus",
        {"target": {"runtime_id": "win:20:0x2"}},
    )

    assert observation.status is ActionStatus.SUCCESS
    assert observation.data["verified"] is True
    assert observation.data["after"]["foreground"] is True
    assert "processes" not in observation.data
    assert "windows" not in observation.data
    assert len(memory.receipts) == 1
    snapshot = activity.snapshot(run_id="windows-run")
    assert any(node.component == "windows" for node in snapshot.nodes)
    assert any(event.action == "windows.window.focus" for event in snapshot.events)


def test_launch_executes_without_repeated_confirmation() -> None:
    backend, _, _, _, _, _, executor = _runtime_stack()
    request = {"application": "Notepad"}

    executed = _execute(executor, "windows.application.launch", request, call_id="launch-1")
    assert executed.status is ActionStatus.SUCCESS
    assert backend.launched == ["notepad.exe"]


def test_normalized_backend_error_remains_actionable_without_stack_trace() -> None:
    _, _, _, _, _, _, executor = _runtime_stack()

    observation = _execute(
        executor,
        "windows.window.focus",
        {"target": {"runtime_id": "win:99:0xff"}},
    )

    assert observation.status is ActionStatus.FAILED
    assert observation.data["error_code"] == "target_stale"
    assert observation.data["verified"] is False
    assert "Traceback" not in observation.message


def test_pre_lora_manifest_and_progressive_schema_disclosure_are_dynamic() -> None:
    _, _, actions, _, _, _, executor = _runtime_stack()
    tools = executor.tool_schemas()
    policy = AgenticPromptPolicy(profile="full", max_relevant_tools=4)

    selected = policy.select_relevant_tools(
        tools,
        "Dime qué aplicación está activa ahorita.",
    )
    context = policy.capability_context(tools)

    assert "windows: AVAILABLE_VIA_TOOL" in context
    assert any(item["function"]["name"] == "windows.window.foreground" for item in selected)
    assert len(actions.list_actions()) == 9


def test_environment_schema_ranks_first_across_action_and_information_paraphrases() -> None:
    _, _, _, _, _, _, executor = _runtime_stack()
    tools = executor.tool_schemas() + [
        {
            "type": "function",
            "function": {
                "name": f"noise.operation_{index}",
                "description": "Unrelated capability.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(20)
    ]
    policy = AgenticPromptPolicy(profile="full", max_relevant_tools=12)
    prompts = (
        "Revisa qué versión de Windows tengo.",
        "Dime qué build de Windows está corriendo esta PC.",
        "Comprueba la arquitectura y versión del sistema operativo.",
        "¿Qué sistema operativo está usando esta máquina? Revísalo ahora.",
        "Revisa los datos básicos de mi Windows.",
        "¿Cómo puedo ver qué versión de Windows tengo?",
    )

    for prompt in prompts:
        selected = policy.select_relevant_tools(tools, prompt)
        assert selected[0]["function"]["name"] == "windows.environment.inspect"
        assert len(selected) < len(tools)


def test_capability_absent_manifest_does_not_advertise_windows() -> None:
    class Payload(BaseModel):
        pass

    definition = ActionDefinition(
        name="calculator.evaluate",
        description="Evaluate arithmetic.",
        payload_model=Payload,
        handler=lambda _context, _payload: ActionResult(message="ok"),
    )
    registry = ActionRegistry()
    registry.register(definition)
    events = EventBus()
    executor = ActionExecutor(
        ActionRouter(registry, Settings(_env_file=None), _Memory(), _Models(), events)
    )

    context = AgenticPromptPolicy.capability_context(executor.tool_schemas())
    assert "windows: AVAILABLE_VIA_TOOL" not in context


def test_action_execution_is_provider_neutral() -> None:
    _, _, _, _, _, _, executor = _runtime_stack()

    # ActionExecutor consumes the provider-neutral tool call contract; no provider
    # identity is supplied or inspected anywhere in this execution path.
    observation = _execute(executor, "windows.environment.inspect")

    assert observation.status is ActionStatus.SUCCESS
    assert observation.action == "windows.environment.inspect"
