from __future__ import annotations

from typing import Any
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from jarvis.action_executor import ActionRequest, ActionStatus
from jarvis.actions.models import ActionResult, ActionRisk, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.terminal_runtime import ExecutionState, TerminalError, TerminalRuntime
from jarvis.tools.models import ToolExecutionStatus, ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class StartSessionPayload(BaseModel):
    shell: str = "powershell"
    cwd: str | None = None
    environment: dict[str, str | None] = Field(default_factory=dict)
    venv_path: str | None = None


class RunTerminalPayload(BaseModel):
    session_id: str
    command: str
    timeout_seconds: float | None = 30.0
    wait: bool = True
    accepts_input: bool = False


class ExecutionPayload(BaseModel):
    session_id: str
    execution_id: str


class ReadOutputPayload(ExecutionPayload):
    after_sequence: int = 0


class StatusPayload(BaseModel):
    session_id: str
    execution_id: str | None = None


class WriteTerminalPayload(ExecutionPayload):
    text: str
    newline: bool = True


class CloseSessionPayload(BaseModel):
    session_id: str


class OpenVisiblePayload(BaseModel):
    shell: str = "powershell"
    cwd: str | None = None
    command: str | None = None


class TerminalModule:
    name = "terminal"
    description = "Persistent, owned PowerShell and CMD terminal sessions."

    def __init__(self, runtime: TerminalRuntime) -> None:
        self.runtime = runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        definitions = (
            ("terminal.start_session", "Start an owned persistent PowerShell or CMD session.", StartSessionPayload, self._start, ActionRisk.SAFE, False, ("terminal session", "open terminal", "start terminal")),
            ("terminal.run", "Run one command in a persistent terminal session; supports background execution.", RunTerminalPayload, self._run, ActionRisk.SENSITIVE, True, ("terminal", "command", "run command", "execute command", "ejecutar comando", "gpu", "hardware", "uso de gpu", "nvidia-smi")),
            ("terminal.write", "Write controlled stdin to an explicitly input-capable running command.", WriteTerminalPayload, self._write, ActionRisk.SENSITIVE, False, ()),
            ("terminal.read_output", "Read ordered incremental stdout and stderr from an execution.", ReadOutputPayload, self._read, ActionRisk.SAFE, False, ()),
            ("terminal.status", "Read terminal session or execution status.", StatusPayload, self._status, ActionRisk.SAFE, False, ()),
            ("terminal.cancel", "Cancel a JARVIS-owned terminal execution and its owned process tree.", ExecutionPayload, self._cancel, ActionRisk.SAFE, True, ()),
            ("terminal.close_session", "Close a JARVIS-owned terminal session.", CloseSessionPayload, self._close, ActionRisk.SAFE, True, ()),
            ("terminal.open_visible", "Explicitly open a visible Windows Terminal window.", OpenVisiblePayload, self._open_visible, ActionRisk.SENSITIVE, False, ()),
        )
        for name, description, payload, handler, risk, cancellation, extra_aliases in definitions:
            aliases = extra_aliases + (("python version", "versión de python", "software version") if name == "terminal.run" else ())
            companions = ("terminal.start_session",) if name == "terminal.run" else ()
            registry.register(ActionDefinition(name=name, description=description, payload_model=payload, handler=handler, tags=("terminal", "shell"), risk=risk, idempotent=False, supports_cancellation=cancellation, semantic_aliases=aliases, companion_names=companions))

    def register_tools(self, registry: ToolRegistry) -> None:
        definitions: tuple[tuple[str, str, type[BaseModel]], ...] = (
            ("terminal.start_session", "Start an owned persistent PowerShell or CMD session.", StartSessionPayload),
            ("terminal.run", "Run a command in a persistent terminal session.", RunTerminalPayload),
            ("terminal.write", "Send stdin to an explicitly input-capable command.", WriteTerminalPayload),
            ("terminal.read_output", "Read incremental terminal output.", ReadOutputPayload),
            ("terminal.status", "Read terminal session or command status.", StatusPayload),
            ("terminal.cancel", "Cancel an owned terminal execution.", ExecutionPayload),
            ("terminal.close_session", "Close an owned terminal session.", CloseSessionPayload),
            ("terminal.open_visible", "Explicitly open Windows Terminal.", OpenVisiblePayload),
        )
        for name, description, payload in definitions:
            registry.register(ToolDefinition(name=name, description=description, input_model=payload, handler=self._tool_handler(name), tags=("terminal",)))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        actions = ("terminal.start_session", "terminal.run", "terminal.write", "terminal.read_output", "terminal.status", "terminal.cancel", "terminal.close_session", "terminal.open_visible")
        registry.register(
            CapabilityDescriptor(
                name="terminal.runtime",
                module_name=self.name,
                intent="operate",
                description="Use persistent PowerShell or CMD sessions and optional visible Windows Terminal.",
                action_names=actions,
                tool_names=actions,
                keywords=("terminal", "powershell", "cmd", "command", "tests"),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            )
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        self.runtime.shutdown()

    def _start(self, context: ActionContext, payload: StartSessionPayload) -> ActionResult:
        if context.dry_run:
            return ActionResult(message="terminal session validated", data=payload.model_dump(mode="json"))
        return self._call(lambda: self.runtime.start_session(payload.shell, cwd=payload.cwd, environment=payload.environment, venv_path=payload.venv_path, **self._runtime_context(context)), "terminal session started")

    def _run(self, context: ActionContext, payload: RunTerminalPayload) -> ActionResult:
        if context.dry_run:
            return ActionResult(message="terminal command validated", data={"session_id": payload.session_id, "timeout_seconds": payload.timeout_seconds})
        return self._call(lambda: self.runtime.run(payload.session_id, payload.command, timeout_seconds=payload.timeout_seconds, wait=payload.wait, accepts_input=payload.accepts_input, **self._runtime_context(context)), "terminal command accepted", execution_result=True)

    def _write(self, context: ActionContext, payload: WriteTerminalPayload) -> ActionResult:
        return self._call(lambda: self.runtime.write(payload.session_id, payload.execution_id, payload.text, newline=payload.newline), "terminal input written")

    def _read(self, context: ActionContext, payload: ReadOutputPayload) -> ActionResult:
        return self._call(lambda: self.runtime.read_output(payload.session_id, payload.execution_id, after_sequence=payload.after_sequence), "terminal output read", execution_result=True)

    def _status(self, context: ActionContext, payload: StatusPayload) -> ActionResult:
        return self._call(lambda: self.runtime.status(payload.session_id, payload.execution_id), "terminal status read")

    def _cancel(self, context: ActionContext, payload: ExecutionPayload) -> ActionResult:
        return self._call(lambda: self.runtime.cancel(payload.session_id, payload.execution_id), "terminal command cancelled", execution_result=True)

    def _close(self, context: ActionContext, payload: CloseSessionPayload) -> ActionResult:
        return self._call(lambda: self.runtime.close_session(payload.session_id), "terminal session closed")

    def _open_visible(self, context: ActionContext, payload: OpenVisiblePayload) -> ActionResult:
        return self._call(
            lambda: self.runtime.open_visible_terminal(
                payload.shell,
                cwd=payload.cwd,
                command=payload.command,
                **self._runtime_context(context),
            ),
            "visible terminal launched",
        )

    @staticmethod
    def _call(operation: Callable[[], Any], message: str, *, execution_result: bool = False) -> ActionResult:
        try:
            value = operation()
        except TerminalError as exc:
            return ActionResult(status=ExecutionStatus.FAILED, message=str(exc), data={"error_code": exc.code.value})
        data = value.model_dump(mode="json") if isinstance(value, BaseModel) else {}
        failed = execution_result and data.get("status") in {ExecutionState.FAILED.value, ExecutionState.TIMED_OUT.value, ExecutionState.CANCELLED.value}
        return ActionResult(status=ExecutionStatus.FAILED if failed else ExecutionStatus.SUCCESS, message=message, data=data)

    @staticmethod
    def _runtime_context(context: ActionContext) -> dict[str, str]:
        run_id = context.metadata.get("agent_run_id")
        return {
            **({"run_id": run_id} if isinstance(run_id, str) and run_id else {}),
            **({"correlation_id": context.correlation_id} if context.correlation_id else {}),
        }

    @staticmethod
    def _tool_handler(action_name: str):
        def handler(context: ToolContext, payload: BaseModel) -> ToolResult:
            if context.dry_run:
                receipt = context.action_router.execute(
                    action_name,
                    payload.model_dump(mode="json"),
                    correlation_id=context.correlation_id,
                    dry_run=True,
                    metadata={**context.metadata, "invoked_via": "tool_registry"},
                )
                return ToolResult(message=receipt.message, data=receipt.data)
            if context.action_executor is None:
                return ToolResult(status=ToolExecutionStatus.FAILED, message="secured action executor is unavailable", data={"error_code": "SECURITY_BLOCKED"})
            observation = context.action_executor.execute(
                ActionRequest(
                    run_id=context.correlation_id,
                    call_id=uuid4().hex,
                    action=action_name,
                    arguments=payload.model_dump(mode="json"),
                    correlation_id=context.correlation_id,
                    metadata={**context.metadata, "invoked_via": "tool_registry"},
                )
            )
            success = observation.status is ActionStatus.SUCCESS
            security_data = (
                {"error_code": "SECURITY_BLOCKED"}
                if observation.status is ActionStatus.BLOCKED
                else {}
            )
            return ToolResult(
                status=(
                    ToolExecutionStatus.SUCCESS
                    if success
                    else ToolExecutionStatus.FAILED
                ),
                message=observation.message,
                data={
                    **observation.data,
                    **security_data,
                    **(
                        {"confirmation_id": observation.confirmation_id}
                        if observation.confirmation_id
                        else {}
                    ),
                },
            )

        return handler
