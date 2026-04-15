from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.indexing_runtime.models import IndexRunRequest, IndexSourceCreateRequest


def install_indexing_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/index/status")
    def index_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.indexing_status()

    @app.post("/index/run")
    def index_run(body: IndexRunRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.indexing_run(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/index/source")
    def index_source(body: IndexSourceCreateRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.indexing_add_source(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/index/reindex")
    def index_reindex(body: IndexRunRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.indexing_reindex(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
