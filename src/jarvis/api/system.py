from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.services import summarize_system_operation, summarize_system_search
from jarvis.system_runtime.base import SystemOpenRequest, SystemResolveRequest, SystemSearchRequest


def install_system_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/system/status")
    def system_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.system_status()

    @app.post("/system/search")
    def system_search(body: SystemSearchRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.system_search(body).model_dump(mode="json")
            return {**receipt, "summary": summarize_system_search(receipt)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/system/resolve")
    def system_resolve(body: SystemResolveRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.system_resolve(body).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/system/open")
    def system_open(body: SystemOpenRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.system_open(body).model_dump(mode="json")
            return {**receipt, "summary": summarize_system_operation(receipt)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/system/open/path")
    def system_open_path(body: SystemOpenRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.system_open_path(
                body.path or "",
                reveal_in_folder=body.reveal_in_folder,
                dry_run=body.dry_run,
                metadata=body.metadata,
            ).model_dump(mode="json")
            return {**receipt, "summary": summarize_system_operation(receipt)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/system/open/app")
    def system_open_app(body: SystemOpenRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.system_open_application(
                body.application or "",
                dry_run=body.dry_run,
                metadata=body.metadata,
            ).model_dump(mode="json")
            return {**receipt, "summary": summarize_system_operation(receipt)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/system/reveal")
    def system_reveal(body: SystemOpenRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = get_jarvis(request).runtime_service.system_reveal(
                body.path or "",
                dry_run=body.dry_run,
                metadata=body.metadata,
            ).model_dump(mode="json")
            return {**receipt, "summary": summarize_system_operation(receipt)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
