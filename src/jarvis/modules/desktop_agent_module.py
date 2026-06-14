from __future__ import annotations

from pydantic import BaseModel

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode


class DesktopAgentRunPayload(BaseModel):
    goal: str
    wait_for_completion: bool = True
    max_steps: int | None = None
    max_retries_per_step: int | None = None
    verbose_trace: bool = False


class DesktopAgentModule:
    name = "desktop_agent_runtime"
    description = "Agent-first desktop runtime for grounded observe-plan-act-verify-recover missions."

    def __init__(self, desktop_agent_runtime_service_factory) -> None:
        self._desktop_agent_runtime_service_factory = desktop_agent_runtime_service_factory

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="desktop_agent.run_mission",
                description="Execute a grounded desktop mission through the observe-plan-act-verify-recover loop.",
                payload_model=DesktopAgentRunPayload,
                handler=self._run_mission,
                tags=("desktop-agent", "desktop", "automation"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="desktop_agent.desktop_operate",
                module_name=self.name,
                intent="desktop_agent",
                description="Operate the desktop through a persistent grounded agent runtime instead of single-shot chat or tool calls.",
                action_names=("desktop_agent.run_mission",),
                keywords=(
                    "abre chrome",
                    "abre word",
                    "abre vscode",
                    "haz click",
                    "haz clic",
                    "click en",
                    "llena este formulario",
                    "ve a la ventana activa",
                    "que hay en mi pantalla",
                    "que ves en mi escritorio",
                    "guarda el documento",
                ),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
                supports_planning=False,
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _run_mission(self, context: ActionContext, payload: DesktopAgentRunPayload) -> ActionResult:
        receipt = self._desktop_agent_runtime_service_factory().run(
            {
                "goal": payload.goal,
                "wait_for_completion": payload.wait_for_completion,
                "max_steps": payload.max_steps,
                "max_retries_per_step": payload.max_retries_per_step,
                "verbose_trace": payload.verbose_trace,
                "metadata": context.metadata,
            }
        )
        return ActionResult(message=receipt.summary, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("goal", request.query or "")
        payload.setdefault("wait_for_completion", True)
        return [ActionStep(action="desktop_agent.run_mission", payload=payload)]
