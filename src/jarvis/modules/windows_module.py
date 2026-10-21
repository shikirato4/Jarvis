from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionRisk, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.windows_runtime import (
    WindowTarget,
    WindowsRuntimeError,
    WindowsRuntimeService,
)


class WindowsEnvironmentPayload(BaseModel):
    pass


class WindowsProcessListPayload(BaseModel):
    limit: int = Field(default=30, ge=1, le=100)
    interactive_only: bool = Field(
        default=False,
        description="When true, return only processes with a visible top-level application window.",
    )


class WindowsWindowListPayload(BaseModel):
    limit: int = Field(default=30, ge=1, le=100)


class WindowsWindowActionPayload(BaseModel):
    target: WindowTarget
    timeout_seconds: float = Field(default=2.0, ge=0.1, le=10.0)


class WindowsApplicationLaunchPayload(BaseModel):
    application: str = Field(
        min_length=1,
        max_length=80,
        description="Semantic name from the safe application catalog; never a command or path.",
    )
    timeout_seconds: float = Field(default=5.0, ge=0.1, le=15.0)


class WindowsModule:
    name = "windows_runtime"
    description = "Provider-neutral semantic Windows inspection and reversible control."

    def __init__(self, runtime: WindowsRuntimeService) -> None:
        self._runtime = runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        semantics = {
            "windows.environment.inspect": (("windows version", "operating system version", "version de windows", "versión de windows", "sistema operativo", "edición de windows"), ()),
            "windows.application.launch": (("open", "launch app", "start app", "abrir aplicación", "abre", "iniciar programa", "inicia aplicación", "inicia"), ("windows.window.list",)),
            "windows.window.list": (("list windows", "show open windows", "listar ventanas", "ventanas abiertas"), ("windows.window.focus",)),
            "windows.window.focus": (("focus window", "activate window", "enfocar ventana", "activar ventana"), ("windows.window.list",)),
        }
        definitions = (
            ("windows.environment.inspect", "Read the current machine's real operating-system facts (sistema operativo): Windows edition, release/version (versión), build/compilación, architecture/arquitectura, basic system details (datos básicos), and session state. Use when the user asks JARVIS to check, inspect, review, verify, or report this PC's actual OS details now; never infer current-device facts from general knowledge. Do not execute for a how-to or explanation of ways the user can check those details unless they also request inspection now. Returns a bounded verified observation without environment variables or credentials.", WindowsEnvironmentPayload, self._environment, ActionRisk.SAFE, True),
            ("windows.process.list", "List a bounded set of Windows processes (procesos) with interactive-application relevance; returns no command-line arguments.", WindowsProcessListPayload, self._processes, ActionRisk.SAFE, True),
            ("windows.window.list", "Enumerate bounded user-visible top-level Windows application windows (ventanas abiertas) with structured runtime identities and state.", WindowsWindowListPayload, self._windows, ActionRisk.SAFE, True),
            ("windows.window.foreground", "Identify the current foreground/active Windows application window (ventana o aplicación activa) from real OS state.", WindowsEnvironmentPayload, self._foreground, ActionRisk.SAFE, True),
            ("windows.window.focus", "Resolve one semantic Windows target, focus/enfocar it, and verify it became the foreground window; ambiguous targets are rejected.", WindowsWindowActionPayload, self._focus, ActionRisk.SAFE, True),
            ("windows.window.minimize", "Minimize/minimizar one resolved Windows application window and verify the resulting state. Safe when already minimized.", WindowsWindowActionPayload, self._minimize, ActionRisk.SAFE, True),
            ("windows.window.maximize", "Maximize/maximizar one resolved Windows application window and verify the resulting state. Safe when already maximized.", WindowsWindowActionPayload, self._maximize, ActionRisk.SAFE, True),
            ("windows.window.restore", "Restore/restaurar one resolved Windows application window to its normal non-minimized and non-maximized size, and verify both state flags are false. Use this action when the target is minimized OR maximized; foreground/visible alone does not mean it is already restored. Safe when already in normal state.", WindowsWindowActionPayload, self._restore, ActionRisk.SAFE, True),
            ("windows.application.launch", "Launch/open (abrir) one application from a fixed safe Windows catalog and verify a new application window; arbitrary paths and shell commands are rejected.", WindowsApplicationLaunchPayload, self._launch, ActionRisk.SENSITIVE, False),
        )
        for name, description, payload, handler, risk, idempotent in definitions:
            aliases, companions = semantics.get(name, ((), ()))
            registry.register(
                ActionDefinition(
                    name=name,
                    description=description,
                    payload_model=payload,
                    handler=handler,
                    tags=("windows", "native", "inspection" if risk is ActionRisk.SAFE and ".inspect" in name or name.endswith(".list") or name.endswith(".foreground") else "control"),
                    risk=risk,
                    idempotent=idempotent,
                    supports_cancellation=True,
                    adapter="windows_native",
                    semantic_aliases=aliases,
                    companion_names=companions,
                )
            )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        names = (
            "windows.environment.inspect",
            "windows.process.list",
            "windows.window.list",
            "windows.window.foreground",
            "windows.window.focus",
            "windows.window.minimize",
            "windows.window.maximize",
            "windows.window.restore",
            "windows.application.launch",
        )
        registry.register(
            CapabilityDescriptor(
                name="windows.native_control",
                module_name=self.name,
                intent="windows",
                description="Inspect real Windows state and perform bounded semantic window/application actions through SecurityGate and ActionExecutor.",
                action_names=names,
                tool_names=names,
                keywords=(
                    "windows",
                    "ventanas",
                    "ventana",
                    "aplicación activa",
                    "aplicacion activa",
                    "foreground",
                    "procesos",
                    "enfoca",
                    "minimiza",
                    "maximiza",
                    "restaura",
                    "bloc de notas",
                    "notepad",
                ),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            )
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _environment(self, _context: ActionContext, _payload: WindowsEnvironmentPayload) -> ActionResult:
        return self._result(self._runtime.inspect_environment)

    def _processes(self, _context: ActionContext, payload: WindowsProcessListPayload) -> ActionResult:
        return self._result(self._runtime.list_processes, limit=payload.limit, interactive_only=payload.interactive_only)

    def _windows(self, _context: ActionContext, payload: WindowsWindowListPayload) -> ActionResult:
        return self._result(self._runtime.list_windows, limit=payload.limit)

    def _foreground(self, _context: ActionContext, _payload: WindowsEnvironmentPayload) -> ActionResult:
        return self._result(self._runtime.foreground_window)

    def _focus(self, _context: ActionContext, payload: WindowsWindowActionPayload) -> ActionResult:
        return self._result(self._runtime.focus_window, payload.target, timeout_seconds=payload.timeout_seconds)

    def _minimize(self, _context: ActionContext, payload: WindowsWindowActionPayload) -> ActionResult:
        return self._result(self._runtime.minimize_window, payload.target, timeout_seconds=payload.timeout_seconds)

    def _maximize(self, _context: ActionContext, payload: WindowsWindowActionPayload) -> ActionResult:
        return self._result(self._runtime.maximize_window, payload.target, timeout_seconds=payload.timeout_seconds)

    def _restore(self, _context: ActionContext, payload: WindowsWindowActionPayload) -> ActionResult:
        return self._result(self._runtime.restore_window, payload.target, timeout_seconds=payload.timeout_seconds)

    def _launch(self, _context: ActionContext, payload: WindowsApplicationLaunchPayload) -> ActionResult:
        return self._result(self._runtime.launch_application, payload.application, timeout_seconds=payload.timeout_seconds)

    @staticmethod
    def _result(operation, *args, **kwargs) -> ActionResult:
        try:
            observation = operation(*args, **kwargs)
        except WindowsRuntimeError as exc:
            return ActionResult(
                status=ExecutionStatus.FAILED,
                message=exc.safe_message,
                data={
                    "status": "error",
                    "verified": False,
                    "error_code": exc.code.value,
                    "candidates": [item.model_dump(mode="json") for item in exc.candidates],
                },
            )
        return ActionResult(
            message=observation.message or f"Windows {observation.operation} completed and verified.",
            data=observation.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            ),
        )
