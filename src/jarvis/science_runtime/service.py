from __future__ import annotations

from jarvis.core.models import HealthStatus, ServiceStatus

from .base import ScienceResult, ScienceSimulationRequest, ScienceSolveRequest
from .simulation import run_simulation
from .solver import solve_problem


class ScienceRuntimeService:
    service_name = "science_runtime"

    def __init__(self, settings, *, logger=None) -> None:
        self._settings = settings
        self._logger = logger
        self._started = False
        self._artifacts_dir = (settings.resolved_data_dir / "science").resolve()

    def start(self) -> None:
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.service_name,
            status=HealthStatus.READY if self._started else HealthStatus.STOPPED,
            details=self.status(),
        )

    def status(self) -> dict[str, object]:
        return {
            "started": self._started,
            "artifacts_dir": str(self._artifacts_dir),
            "capabilities": [
                "symbolic_math",
                "physics_estimation",
                "numerical_simulation",
                "visualization",
            ],
        }

    def solve(self, request: ScienceSolveRequest) -> ScienceResult:
        return solve_problem(request)

    def simulate(self, request: ScienceSimulationRequest) -> ScienceResult:
        capped = request.model_copy(update={"max_points": min(max(request.max_points, 10), 5000)})
        return run_simulation(capped, output_dir=self._artifacts_dir)
