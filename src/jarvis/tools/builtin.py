from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from jarvis.tools.models import ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class MemoryLookupToolPayload(BaseModel):
    query: str
    limit: int = 10


class WorkspaceSearchToolPayload(BaseModel):
    query: str
    limit: int = 10


class DocumentComposeToolPayload(BaseModel):
    title: str
    objective: str
    findings: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    output_path: str | None = None
    persist_memory: bool = True


class ImageInspectToolPayload(BaseModel):
    image_path: str


class AudioInspectToolPayload(BaseModel):
    audio_path: str


class ShellCommandToolPayload(BaseModel):
    command: list[str]
    cwd: str | None = None
    timeout_seconds: int = 30


class StateSnapshotToolPayload(BaseModel):
    include_history: bool = True


class ModelChatToolPayload(BaseModel):
    prompt: str
    logical_model: str | None = None
    task_type: str = "assistant"
    temperature: float | None = None
    timeout_seconds: float | None = None


def install_builtin_tools(registry: ToolRegistry, state_provider: Callable[..., object]) -> None:
    registry.register(
        ToolDefinition(
            name="memory.lookup",
            description="Search persisted memories by text.",
            input_model=MemoryLookupToolPayload,
            handler=_memory_lookup,
            tags=("memory", "state"),
        )
    )
    registry.register(
        ToolDefinition(
            name="workspace.search",
            description="Search workspace files for evidence.",
            input_model=WorkspaceSearchToolPayload,
            handler=_workspace_search,
            tags=("research",),
        )
    )
    registry.register(
        ToolDefinition(
            name="document.compose",
            description="Compose a markdown note or brief.",
            input_model=DocumentComposeToolPayload,
            handler=_document_compose,
            tags=("writer",),
        )
    )
    registry.register(
        ToolDefinition(
            name="image.inspect",
            description="Inspect local image metadata.",
            input_model=ImageInspectToolPayload,
            handler=_image_inspect,
            tags=("vision",),
        )
    )
    registry.register(
        ToolDefinition(
            name="audio.inspect",
            description="Inspect local audio metadata.",
            input_model=AudioInspectToolPayload,
            handler=_audio_inspect,
            tags=("voice",),
        )
    )
    registry.register(
        ToolDefinition(
            name="shell.command",
            description="Run an allowlisted local command through the operations layer.",
            input_model=ShellCommandToolPayload,
            handler=_shell_command,
            tags=("operations",),
        )
    )
    registry.register(
        ToolDefinition(
            name="runtime.snapshot",
            description="Get a current runtime snapshot.",
            input_model=StateSnapshotToolPayload,
            handler=_make_state_snapshot_tool(state_provider),
            tags=("state",),
        )
    )
    registry.register(
        ToolDefinition(
            name="model.chat",
            description="Run direct model inference through the model service.",
            input_model=ModelChatToolPayload,
            handler=_model_chat,
            tags=("models", "reasoning"),
        )
    )


def _memory_lookup(context: ToolContext, payload: MemoryLookupToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "memory.search",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data)


def _workspace_search(context: ToolContext, payload: WorkspaceSearchToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "research.workspace_search",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data)


def _document_compose(context: ToolContext, payload: DocumentComposeToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "writer.compose_note",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data, artifacts=receipt.artifacts)


def _image_inspect(context: ToolContext, payload: ImageInspectToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "vision.inspect_image",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data)


def _audio_inspect(context: ToolContext, payload: AudioInspectToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "voice.inspect_audio",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data)


def _shell_command(context: ToolContext, payload: ShellCommandToolPayload) -> ToolResult:
    receipt = context.action_router.execute(
        "operations.run_command",
        payload.model_dump(mode="json"),
        correlation_id=context.correlation_id,
        dry_run=context.dry_run,
        metadata={**context.metadata, "invoked_via": "tool"},
    )
    return ToolResult(message=receipt.message, data=receipt.data)


def _make_state_snapshot_tool(state_provider: Callable[..., object]):
    def _handler(context: ToolContext, payload: StateSnapshotToolPayload) -> ToolResult:
        snapshot = state_provider(include_history=payload.include_history)
        return ToolResult(message="runtime snapshot created", data=snapshot.model_dump(mode="json"))

    return _handler


def _model_chat(context: ToolContext, payload: ModelChatToolPayload) -> ToolResult:
    from jarvis.models_runtime.base import ModelMessage, ModelRequest

    response = context.models.infer(
        ModelRequest(
            task_type=payload.task_type,
            logical_model=payload.logical_model,
            prompt=payload.prompt,
            messages=[ModelMessage(role="user", content=payload.prompt)],
            temperature=payload.temperature,
            timeout_seconds=payload.timeout_seconds,
            correlation_id=context.correlation_id,
            metadata=context.metadata,
        )
    )
    return ToolResult(
        message="model inference completed",
        data=response.model_dump(mode="json"),
    )
