from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request
from pydantic import Field

from jarvis.core.errors import JarvisError
from jarvis.models.base import JarvisBaseModel


class BreakerResetRequest(JarvisBaseModel):
    service_name: str
    dependency_name: str | None = None


class RecoveryRequest(JarvisBaseModel):
    dry_run: bool = False


class RetentionSweepRequest(JarvisBaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationCancelRequest(JarvisBaseModel):
    reason: str = "cancel requested from api"


def install_ops_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/ops/status")
    def ops_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_status()

    @app.get("/ops/health")
    def ops_health(request: Request) -> dict[str, Any]:
        probes = get_jarvis(request).runtime_service.ops_health()
        return {"probes": [probe.model_dump(mode="json") for probe in probes]}

    @app.get("/ops/snapshot")
    def ops_snapshot(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_snapshot().model_dump(mode="json")

    @app.get("/ops/operations")
    def ops_operations(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_operations()

    @app.get("/ops/resources")
    def ops_resources(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_resources()

    @app.get("/ops/diagnostics")
    def ops_diagnostics(request: Request) -> dict[str, Any]:
        reports = get_jarvis(request).runtime_service.ops_diagnostics()
        return {"reports": [report.model_dump(mode="json") for report in reports]}

    @app.get("/ops/diagnostics/{service_name}")
    def ops_diagnostics_service(service_name: str, request: Request) -> dict[str, Any]:
        reports = get_jarvis(request).runtime_service.ops_diagnostics(service_name)
        return {"reports": [report.model_dump(mode="json") for report in reports]}

    @app.post("/ops/recover/{service_name}")
    def ops_recover(service_name: str, body: RecoveryRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.ops_recover_service(service_name, dry_run=body.dry_run).model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/ops/breakers/reset")
    def ops_reset_breaker(body: BreakerResetRequest, request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_reset_breaker(body.service_name, body.dependency_name)

    @app.post("/ops/retention/sweep")
    def ops_retention_sweep(body: RetentionSweepRequest, request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_retention_sweep().model_dump(mode="json")

    @app.post("/ops/operations/{operation_id}/cancel")
    def ops_cancel_operation(operation_id: str, body: OperationCancelRequest, request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.ops_cancel_operation(operation_id, reason=body.reason)
