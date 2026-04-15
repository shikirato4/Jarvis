from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import Field

from jarvis.api.autonomy import install_autonomy_routes
from jarvis.api.desktop_agent import install_desktop_agent_routes
from jarvis.api.hud import install_hud_routes
from jarvis.api.indexing import install_indexing_routes
from jarvis.api.ops import install_ops_routes
from jarvis.api.research import install_research_routes
from jarvis.api.science import install_science_routes
from jarvis.api.security import install_security_routes
from jarvis.api.semantic import install_semantic_routes
from jarvis.api.system import install_system_routes
from jarvis.api.unity import install_unity_routes
from jarvis.api.ui import install_ui_routes
from jarvis.api.vision import install_vision_routes
from jarvis.api.voice import install_voice_routes
from jarvis.api.writing import install_writing_routes
from jarvis.automation.service import AutomationDefinition
from jarvis.bootstrap import JarvisApplication, build_application
from jarvis.cognition.models import OrchestrationRequest
from jarvis.core.modes import ExecutionMode
from jarvis.core.errors import JarvisError
from jarvis.models_runtime.base import ModelRequest
from jarvis.models.base import JarvisBaseModel
from jarvis.routing.models import TaskRequest


class ActionExecutionRequest(JarvisBaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionRequest(JarvisBaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModeChangeRequest(JarvisBaseModel):
    mode: ExecutionMode
    reason: str | None = None
    sticky: bool = True


class ModelInferenceRequest(JarvisBaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    logical_model: str | None = None
    task_type: str = "assistant"
    temperature: float | None = None
    timeout_seconds: float | None = None
    required_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    jarvis = build_application()
    jarvis.start()
    app.state.jarvis = jarvis
    try:
        yield
    finally:
        jarvis.stop()


def create_api_app() -> FastAPI:
    app = FastAPI(title="Jarvis Cognitive OS", version="0.1.0", lifespan=_lifespan)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        snapshot = jarvis.runtime_service.snapshot()
        return {
            "status": "ok",
            "app_name": snapshot.app_name,
            "mode": snapshot.mode.model_dump(mode="json"),
            "services": [service.model_dump(mode="json") for service in snapshot.services],
            "actions": len(snapshot.action_names),
            "tools": len(snapshot.tool_names),
        }

    @app.get("/actions")
    def list_actions(request: Request) -> dict[str, Any]:
        return {"actions": _get_jarvis(request).describe()["actions"]}

    @app.get("/tools")
    def list_tools(request: Request) -> dict[str, Any]:
        return {"tools": _get_jarvis(request).describe()["tools"]}

    @app.get("/models")
    def list_models(request: Request) -> dict[str, Any]:
        return {"models": _get_jarvis(request).runtime_service.list_models()}

    @app.get("/providers")
    def provider_health(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        return {
            "providers": [entry.model_dump(mode="json") for entry in jarvis.runtime_service.model_health()],
            "embedding_providers": [entry.model_dump(mode="json") for entry in jarvis.runtime_service.embedding_health()],
        }

    @app.get("/state")
    def state(request: Request) -> dict[str, Any]:
        return _get_jarvis(request).runtime_service.snapshot().model_dump(mode="json")

    @app.get("/mode")
    def mode(request: Request) -> dict[str, Any]:
        return _get_jarvis(request).mode_manager.snapshot().model_dump(mode="json")

    @app.post("/mode")
    def change_mode(body: ModeChangeRequest, request: Request) -> dict[str, Any]:
        try:
            snapshot = _get_jarvis(request).runtime_service.switch_mode(body.mode, reason=body.reason, sticky=body.sticky)
            return snapshot.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/actions/{action_name}")
    def execute_action(action_name: str, body: ActionExecutionRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = _get_jarvis(request).runtime_service.execute_action(
                action_name,
                body.payload,
                dry_run=body.dry_run,
                metadata=body.metadata,
            )
            return receipt.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/tools/{tool_name}")
    def invoke_tool(tool_name: str, body: ToolExecutionRequest, request: Request) -> dict[str, Any]:
        try:
            receipt = _get_jarvis(request).runtime_service.invoke_tool(
                tool_name,
                body.payload,
                dry_run=body.dry_run,
                metadata=body.metadata,
            )
            return receipt.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/tasks")
    def route_task(body: TaskRequest, request: Request) -> dict[str, Any]:
        try:
            response = _get_jarvis(request).runtime_service.route(body)
            return response.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/infer")
    def infer_model(body: ModelInferenceRequest, request: Request) -> dict[str, Any]:
        try:
            response = _get_jarvis(request).runtime_service.infer_model(
                ModelRequest(
                    prompt=body.prompt,
                    messages=body.messages,
                    logical_model=body.logical_model,
                    task_type=body.task_type,
                    temperature=body.temperature,
                    timeout_seconds=body.timeout_seconds,
                    required_capabilities=body.required_capabilities,
                    metadata=body.metadata,
                )
            )
            return response.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/metacommands")
    def run_metacommand(body: TaskRequest, request: Request) -> dict[str, Any]:
        try:
            response = _get_jarvis(request).runtime_service.route(body)
            return response.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/orchestrate")
    def orchestrate(body: OrchestrationRequest, request: Request) -> dict[str, Any]:
        try:
            response = _get_jarvis(request).orchestrator.handle(body)
            return response.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.get("/memory/search")
    def search_memory(request: Request, query: str = Query(...), limit: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
        matches = _get_jarvis(request).memory_service.search_memories(query, limit)
        return {"matches": [entry.model_dump(mode="json") for entry in matches], "count": len(matches)}

    @app.get("/automation")
    def list_automations(request: Request, enabled_only: bool = False) -> dict[str, Any]:
        items = _get_jarvis(request).automation_service.list(enabled_only=enabled_only)
        return {"automations": [entry.model_dump(mode="json") for entry in items]}

    @app.post("/automation")
    def save_automation(body: AutomationDefinition, request: Request) -> dict[str, Any]:
        try:
            entry = _get_jarvis(request).runtime_service.save_automation(body)
            return entry.model_dump(mode="json")
        except (JarvisError, ValueError) as exc:
            if isinstance(exc, JarvisError):
                raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    install_semantic_routes(app, _get_jarvis)
    install_indexing_routes(app, _get_jarvis)
    install_research_routes(app, _get_jarvis)
    install_science_routes(app, _get_jarvis)
    install_security_routes(app, _get_jarvis)
    install_system_routes(app, _get_jarvis)
    install_unity_routes(app, _get_jarvis)
    install_ui_routes(app, _get_jarvis)
    install_vision_routes(app, _get_jarvis)
    install_voice_routes(app, _get_jarvis)
    install_writing_routes(app, _get_jarvis)
    install_autonomy_routes(app, _get_jarvis)
    install_desktop_agent_routes(app, _get_jarvis)
    install_ops_routes(app, _get_jarvis)
    install_hud_routes(app, _get_jarvis)

    return app


def _get_jarvis(request: Request) -> JarvisApplication:
    return request.app.state.jarvis
