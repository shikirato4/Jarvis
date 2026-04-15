from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.science_runtime import ScienceSimulationRequest, ScienceSolveRequest
from jarvis.services import summarize_science_result


def install_science_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/science/status")
    def science_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.science_status()

    @app.post("/science/solve")
    def science_solve(body: ScienceSolveRequest, request: Request) -> dict[str, Any]:
        try:
            result = get_jarvis(request).runtime_service.science_solve(body).model_dump(mode="json")
            return {**result, "summary": summarize_science_result(result)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/science/simulate")
    def science_simulate(body: ScienceSimulationRequest, request: Request) -> dict[str, Any]:
        try:
            result = get_jarvis(request).runtime_service.science_simulate(body).model_dump(mode="json")
            return {**result, "summary": summarize_science_result(result)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
