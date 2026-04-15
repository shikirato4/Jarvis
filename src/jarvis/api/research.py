from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Query, Request

from jarvis.core.errors import JarvisError
from jarvis.research_runtime.models import ResearchRunRequest
from jarvis.services import summarize_research_report, summarize_research_task


def install_research_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/research/status")
    def research_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.research_status()

    @app.post("/research/run")
    def research_run(body: ResearchRunRequest, request: Request) -> dict[str, Any]:
        try:
            task = get_jarvis(request).runtime_service.research_run(body).model_dump(mode="json")
            return {**task, "summary": summarize_research_task(task)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.get("/research/report")
    def research_report(request: Request, task_id: str | None = Query(None)) -> dict[str, Any]:
        try:
            report = get_jarvis(request).runtime_service.research_report(task_id).model_dump(mode="json")
            return {**report, "summary": summarize_research_report(report)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
