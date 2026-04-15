from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.process import ProcessRequest, ProcessRunner
from jarvis.core.safety import ensure_allowed_executable, ensure_within_roots


class RunCommandPayload(BaseModel):
    command: list[str] = Field(min_length=1)
    cwd: str | None = None
    timeout_seconds: int = 30


class OperationsModule:
    name = "operations"
    description = "Validated local operational control and command execution."

    def __init__(self, process_runner: ProcessRunner) -> None:
        self._process_runner = process_runner

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="operations.run_command",
                description="Run an allowlisted local command inside the workspace root.",
                payload_model=RunCommandPayload,
                handler=self._run_command,
                tags=("operations", "shell"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="operations.command",
                module_name=self.name,
                intent="operate",
                description="Execute allowlisted local commands.",
                action_names=("operations.run_command",),
                tool_names=("shell.command",),
                keywords=("comando", "ejecuta", "run", "shell"),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _run_command(self, context: ActionContext, payload: RunCommandPayload) -> ActionResult:
        executable = payload.command[0]
        ensure_allowed_executable(executable, context.settings.command_allowlist)
        cwd = (
            ensure_within_roots(payload.cwd, (context.settings.resolved_workspace_root,), "command execution")
            if payload.cwd
            else context.settings.resolved_workspace_root
        )
        if context.dry_run:
            return ActionResult(
                message="command validated in dry-run mode",
                data={"command": payload.command, "cwd": str(cwd), "timeout_seconds": payload.timeout_seconds},
            )
        completed = self._process_runner.run(
            ProcessRequest(
                command=payload.command,
                cwd=str(cwd),
                timeout_seconds=payload.timeout_seconds,
            )
        )
        return ActionResult(
            message="command executed",
            data={
                "command": completed.command,
                "cwd": completed.cwd,
                "exit_code": completed.exit_code,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        return [ActionStep(action="operations.run_command", payload=dict(request.payload))]
