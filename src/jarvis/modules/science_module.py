from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.science_runtime import ScienceSimulationRequest, ScienceSolveRequest


class ScienceSolvePayload(BaseModel):
    query: str
    operation: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    generate_plot: bool = False


class ScienceSimulationPayload(BaseModel):
    query: str | None = None
    simulation_type: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    duration: float | None = None
    time_step: float | None = None
    max_points: int = 2000
    generate_plot: bool = True


class ScienceModule:
    name = "science"
    description = "Scientific computing runtime for math, physics and simulation."

    def __init__(self, science_runtime) -> None:
        self._science = science_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="science.solve_problem",
                description="Solve symbolic mathematics and scientific estimation tasks.",
                payload_model=ScienceSolvePayload,
                handler=self._solve_problem,
                tags=("science", "mathematics", "physics"),
            )
        )
        registry.register(
            ActionDefinition(
                name="science.run_simulation",
                description="Run numerical simulations and optionally generate plots.",
                payload_model=ScienceSimulationPayload,
                handler=self._run_simulation,
                tags=("science", "simulation", "physics"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="science.runtime",
                module_name=self.name,
                intent="science",
                description="Scientific computation, mathematics and physical simulation.",
                action_names=("science.solve_problem", "science.run_simulation"),
                keywords=(
                    "calcula",
                    "simula",
                    "estima",
                    "resuelve",
                    "que pasaria si",
                    "qué pasaría si",
                    "derivada",
                    "integral",
                    "matriz",
                    "probabilidad",
                    "fisica",
                    "física",
                    "orbita",
                    "órbita",
                    "agujero negro",
                    "dilatacion temporal",
                    "dilatación temporal",
                ),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="analysis",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _solve_problem(self, context: ActionContext, payload: ScienceSolvePayload) -> ActionResult:
        result = self._science.solve(
            ScienceSolveRequest(
                query=payload.query,
                operation=payload.operation,
                parameters=payload.parameters,
                generate_plot=payload.generate_plot,
            )
        )
        return ActionResult(message="science solve completed", data=result.model_dump(mode="json"), artifacts=result.artifacts)

    def _run_simulation(self, context: ActionContext, payload: ScienceSimulationPayload) -> ActionResult:
        result = self._science.simulate(
            ScienceSimulationRequest(
                query=payload.query,
                simulation_type=payload.simulation_type,
                parameters=payload.parameters,
                duration=payload.duration,
                time_step=payload.time_step,
                max_points=payload.max_points,
                generate_plot=payload.generate_plot,
            )
        )
        return ActionResult(message="science simulation completed", data=result.model_dump(mode="json"), artifacts=result.artifacts)

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        lowered = (request.query or "").casefold()
        payload = dict(request.payload)
        if any(token in lowered for token in ("simula", "simul", "orbita", "órbita", "movimiento", "caida", "caída")):
            payload.setdefault("query", request.query)
            return [ActionStep(action="science.run_simulation", payload=payload)]
        payload.setdefault("query", request.query or "")
        return [ActionStep(action="science.solve_problem", payload=payload)]
