from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request

from jarvis.core.errors import JarvisError
from jarvis.security_runtime import SecurityAnalyzeRequest, SecurityPasswordCheckRequest
from jarvis.services import summarize_security_result


def install_security_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/security/status")
    def security_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.security_status()

    @app.post("/security/analyze")
    def security_analyze(body: SecurityAnalyzeRequest, request: Request) -> dict[str, Any]:
        try:
            result = get_jarvis(request).runtime_service.security_analyze(body).model_dump(mode="json")
            return {**result, "summary": summarize_security_result(result)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/security/check-password")
    def security_check_password(body: SecurityPasswordCheckRequest, request: Request) -> dict[str, Any]:
        try:
            result = get_jarvis(request).runtime_service.security_check_password(body).model_dump(mode="json")
            return {**result, "summary": summarize_security_result(result)}
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
