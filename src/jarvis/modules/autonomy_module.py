from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.autonomy.base import MissionControlRequest, MissionPlanRequest, MissionRequest
from jarvis.autonomy.service import AutonomyService
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.tools.models import ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class AutonomyGoalPayload(BaseModel):
    goal: str
    payload: dict[str, object] = Field(default_factory=dict)
    autonomy_level: str | None = None


class AutonomyModule:
    name = "autonomy"
    description = "Autonomous multi-step missions with planning, execution, verification and reflection."

    def __init__(self, autonomy_service: AutonomyService) -> None:
        self._autonomy = autonomy_service

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="autonomy.plan_mission", description="Build an autonomous mission plan.", payload_model=AutonomyGoalPayload, handler=self._plan_mission, tags=("autonomy", "planning")))
        registry.register(ActionDefinition(name="autonomy.start_mission", description="Start an autonomous mission.", payload_model=AutonomyGoalPayload, handler=self._start_mission, tags=("autonomy", "execution")))
        registry.register(ActionDefinition(name="autonomy.step_mission", description="Advance a mission by one step.", payload_model=MissionControlRequest, handler=self._step_mission, tags=("autonomy", "execution")))
        registry.register(ActionDefinition(name="autonomy.stop_mission", description="Stop an autonomous mission.", payload_model=MissionControlRequest, handler=self._stop_mission, tags=("autonomy", "control")))

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(ToolDefinition(name="autonomy.plan", description="Plan an autonomous mission.", input_model=AutonomyGoalPayload, handler=self._tool_plan, tags=("autonomy", "planning")))
        registry.register(ToolDefinition(name="autonomy.start", description="Start an autonomous mission.", input_model=AutonomyGoalPayload, handler=self._tool_start, tags=("autonomy", "execution")))
        registry.register(ToolDefinition(name="autonomy.inspect", description="Inspect a mission snapshot.", input_model=MissionControlRequest, handler=self._tool_inspect, tags=("autonomy", "state")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="autonomy.agent",
                module_name=self.name,
                intent="autonomous_task",
                description="Run a bounded autonomous mission with planning, execution and verification.",
                action_names=("autonomy.plan_mission", "autonomy.start_mission", "autonomy.step_mission", "autonomy.stop_mission"),
                tool_names=("autonomy.plan", "autonomy.start", "autonomy.inspect"),
                keywords=("autonomo", "autonomous", "mission", "multi-step", "hazlo por tu cuenta"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
                supports_planning=False,
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _plan_mission(self, context: ActionContext, payload: AutonomyGoalPayload) -> ActionResult:
        plan = self._autonomy.plan_mission(MissionPlanRequest(goal=payload.goal, payload=payload.payload, autonomy_level=payload.autonomy_level))
        return ActionResult(message="mission plan created", data=plan.model_dump(mode="json"))

    def _start_mission(self, context: ActionContext, payload: AutonomyGoalPayload) -> ActionResult:
        receipt = self._autonomy.start_mission(MissionRequest(goal=payload.goal, payload=payload.payload, autonomy_level=payload.autonomy_level))
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _step_mission(self, context: ActionContext, payload: MissionControlRequest) -> ActionResult:
        receipt = self._autonomy.step_mission(payload.mission_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _stop_mission(self, context: ActionContext, payload: MissionControlRequest) -> ActionResult:
        receipt = self._autonomy.stop_mission(payload.mission_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_plan(self, context: ToolContext, payload: AutonomyGoalPayload) -> ToolResult:
        plan = self._autonomy.plan_mission(MissionPlanRequest(goal=payload.goal, payload=payload.payload, autonomy_level=payload.autonomy_level))
        return ToolResult(message="mission plan created", data=plan.model_dump(mode="json"))

    def _tool_start(self, context: ToolContext, payload: AutonomyGoalPayload) -> ToolResult:
        receipt = self._autonomy.start_mission(MissionRequest(goal=payload.goal, payload=payload.payload, autonomy_level=payload.autonomy_level))
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_inspect(self, context: ToolContext, payload: MissionControlRequest) -> ToolResult:
        receipt = self._autonomy.inspect_mission(payload.mission_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("goal", request.query or "")
        return [ActionStep(action="autonomy.start_mission", payload=payload)]
