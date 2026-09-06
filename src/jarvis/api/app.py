from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import Field, SecretStr

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
from jarvis.models_runtime.hybrid import LLMMode
from jarvis.models.base import JarvisBaseModel
from jarvis.routing.models import TaskRequest
from jarvis.security_runtime.secrets import SecretStoreUnavailable
from jarvis.local_runtime import LocalLaunchProfile, LocalRuntimeIncompatibleError


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
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    logical_model: str | None = None
    task_type: str = "assistant"
    temperature: float | None = None
    timeout_seconds: float | None = None
    required_capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMModeChangeRequest(JarvisBaseModel):
    mode: LLMMode


class NvidiaKeyRequest(JarvisBaseModel):
    key: SecretStr = Field(repr=False, exclude=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    jarvis = build_application()
    jarvis.start()
    app.state.jarvis = jarvis
    try:
        yield
    finally:
        jarvis.stop()


def _action_catalog(jarvis: Any) -> list[dict[str, Any]]:
    """Return action metadata without probing runtime/provider health."""
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "tags": list(definition.tags),
        }
        for definition in jarvis.action_registry.list_actions()
    ]


def _tool_catalog(jarvis: Any) -> list[dict[str, Any]]:
    """Return tool metadata without probing runtime/provider health."""
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "tags": list(definition.tags),
        }
        for definition in jarvis.tool_registry.list_tools()
    ]


def _model_request(body: ModelInferenceRequest) -> ModelRequest:
    return ModelRequest(
        prompt=body.prompt,
        messages=body.messages,
        tools=body.tools,
        tool_choice=body.tool_choice,
        response_format=body.response_format,
        logical_model=body.logical_model,
        task_type=body.task_type,
        temperature=body.temperature,
        timeout_seconds=body.timeout_seconds,
        required_capabilities=body.required_capabilities,
        metadata=body.metadata,
    )


def _stream_model_events(jarvis: Any, body: ModelInferenceRequest):
    """Encode provider stream chunks as standards-compatible SSE data."""
    try:
        for chunk in jarvis.runtime_service.stream_model(_model_request(body)):
            # Keep the public SSE shape backwards-compatible: tool_calls is
            # emitted only for an actual tool delta, not as an empty list on
            # ordinary text chunks.
            payload = chunk.model_dump(mode="json")
            if not payload.get("tool_calls"):
                payload.pop("tool_calls", None)
            yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    except JarvisError as exc:
        payload = {"text": "", "done": True, "error": exc.code, "metadata": {"recoverable": exc.recoverable}}
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    except Exception as exc:
        payload = {"text": "", "done": True, "error": type(exc).__name__, "metadata": {"recoverable": False}}
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


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
        return {"actions": _action_catalog(_get_jarvis(request))}

    @app.get("/tools")
    def list_tools(request: Request) -> dict[str, Any]:
        return {"tools": _tool_catalog(_get_jarvis(request))}

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

    @app.get("/llm/status")
    def llm_status(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        selection = jarvis.model_service.provider_selection()
        nvidia = jarvis.provider_registry.get("nvidia_nim")
        local = jarvis.provider_registry.get("qwen_local")
        local_capabilities = local.capabilities.model_dump(mode="json") if local else {}
        return {
            "selected_mode": selection.selected_mode.value,
            "active_provider": selection.active_provider,
            "request_provider": selection.request_provider,
            "active_model": selection.active_model,
            "provider_status": selection.provider_status.model_dump(mode="json"),
            "fallback_reason": selection.fallback_reason,
            "persistence_warning": selection.persistence_warning,
            "capabilities": {
                "nvidia_nim": nvidia.capabilities.model_dump(mode="json"),
                "qwen_local": local_capabilities,
                # Deprecated response alias for clients from the Nemotron-local era.
                "nemotron_omni_local": local_capabilities,
            },
            "key_status": jarvis.nvidia_secret_manager.status().model_dump(mode="json"),
            "local_runtime": jarvis.local_model_runtime_manager.status().model_dump(
                mode="json"
            ),
        }

    @app.put("/llm/mode")
    def change_llm_mode(body: LLMModeChangeRequest, request: Request) -> dict[str, Any]:
        _get_jarvis(request).model_service.set_llm_mode(body.mode)
        return llm_status(request)

    @app.put("/llm/nvidia-key")
    def save_nvidia_key(body: NvidiaKeyRequest, request: Request) -> dict[str, Any]:
        try:
            status = _get_jarvis(request).nvidia_secret_manager.save(body.key.get_secret_value())
            return status.model_dump(mode="json")
        except (ValueError, SecretStoreUnavailable) as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    @app.delete("/llm/nvidia-key")
    def delete_nvidia_key(request: Request) -> dict[str, Any]:
        try:
            status = _get_jarvis(request).nvidia_secret_manager.delete()
            return status.model_dump(mode="json")
        except SecretStoreUnavailable as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc

    @app.post("/llm/nvidia-key/test")
    def test_nvidia_key(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        provider = jarvis.provider_registry.get("nvidia_nim")
        status = provider.status_check()
        return {
            "configured": jarvis.nvidia_secret_manager.status().configured,
            "state": status.state.value,
            "healthy": status.healthy,
            "latency_ms": status.latency_ms,
            "message": status.message,
        }

    @app.get("/llm/local-diagnostic")
    def local_llm_diagnostic(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        return jarvis.local_hardware_diagnostics.inspect(
            jarvis.settings.local_llm_base_url
        ).model_dump(mode="json")

    @app.get("/llm/local-runtime")
    def local_llm_runtime(request: Request) -> dict[str, Any]:
        jarvis = _get_jarvis(request)
        diagnostic = jarvis.local_hardware_diagnostics.inspect(
            jarvis.settings.local_llm_base_url
        )
        profile = LocalLaunchProfile.from_settings(jarvis.settings)
        command: tuple[str, ...] | None = None
        launch_error: str | None = None
        try:
            command = jarvis.local_runtime_service.build_launch_command(
                "vllm",
                profile,
                diagnostic,
            )
        except LocalRuntimeIncompatibleError as exc:
            launch_error = str(exc)
        return {
            "diagnostic": diagnostic.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "launch_command": list(command) if command is not None else None,
            "launch_error": launch_error,
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
            response = _get_jarvis(request).runtime_service.infer_model(_model_request(body))
            return response.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/infer/stream")
    def stream_model(body: ModelInferenceRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _stream_model_events(_get_jarvis(request), body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
