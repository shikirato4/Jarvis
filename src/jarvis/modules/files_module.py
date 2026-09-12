from __future__ import annotations

from typing import Any, Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from jarvis.action_executor import ActionRequest, ActionStatus
from jarvis.actions.models import ActionResult, ActionRisk, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.file_runtime import FileNavigator, FileOperationError, OverwritePolicy
from jarvis.tools.models import ToolExecutionStatus, ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class PathPayload(BaseModel):
    path: str


class ResolvePayload(PathPayload):
    base: str | None = None


class ListPayload(BaseModel):
    path: str = "workspace"
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    sort_by: Literal["name", "size", "modified", "type"] = "name"
    kind: Literal["all", "files", "directories"] = "all"
    pattern: str | None = None


class SearchPayload(BaseModel):
    pattern: str
    scope: str = "workspace"
    recursive: bool = True
    max_depth: int = Field(default=8, ge=0, le=32)
    max_results: int = Field(default=50, ge=1, le=100)
    extension: str | None = None
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)


class ContentSearchPayload(BaseModel):
    query: str
    scope: str = "workspace"
    path: str | None = None
    max_results: int = Field(default=30, ge=1, le=100)
    max_bytes: int = Field(default=300_000, ge=1_000, le=1_000_000)
    timeout_seconds: float = Field(default=10.0, gt=0, le=30)


class StatPayload(PathPayload):
    sha256: bool = False


class ReadPayload(PathPayload):
    encoding: str = "utf-8"
    max_bytes: int = Field(default=200_000, ge=1, le=1_000_000)


class CreateDirectoryPayload(PathPayload):
    parents: bool = False
    exist_ok: bool = False


class CreateFilePayload(PathPayload):
    content: str = ""
    encoding: str = "utf-8"
    overwrite: OverwritePolicy = OverwritePolicy.FAIL_IF_EXISTS
    parents: bool = False


class TransferPayload(BaseModel):
    source: str
    destination: str
    overwrite: OverwritePolicy = OverwritePolicy.FAIL_IF_EXISTS


class RenamePayload(PathPayload):
    new_name: str
    overwrite: OverwritePolicy = OverwritePolicy.FAIL_IF_EXISTS


class DeletePayload(PathPayload):
    permanent: bool = False
    recursive: bool = False


class BatchMovePayload(BaseModel):
    items: list[str] = Field(min_length=1, max_length=100)
    destination: str


class FilesModule:
    name = "files"
    description = "Structured, bounded and verified Windows filesystem operations."

    def __init__(self, runtime: FileNavigator) -> None:
        self.runtime = runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        definitions: tuple[tuple[str, str, type[BaseModel], Callable[..., dict[str, Any]], ActionRisk], ...] = (
            ("files.resolve_path", "Resolve a Windows file or folder path, workspace path, or known folder such as Downloads/Descargas.", ResolvePayload, lambda p: {"path": str(self.runtime.resolve_path(p.path, base=p.base))}, ActionRisk.SAFE),
            ("files.list", "List files and folders (archivos y carpetas) with bounded pagination and metadata.", ListPayload, lambda p: self.runtime.list_directory(**p.model_dump()), ActionRisk.SAFE),
            ("files.search", "Search/buscar a file or folder by filename, extension or safe pattern inside a bounded scope.", SearchPayload, lambda p: self.runtime.search(**p.model_dump()), ActionRisk.SAFE),
            ("files.search_content", "Search semantically and lexically inside bounded local text files, returning matched excerpts with provenance.", ContentSearchPayload, lambda p: self.runtime.search_content(**p.model_dump()), ActionRisk.SAFE),
            ("files.stat", "Inspect file/folder size, type, timestamps, attributes and optional SHA-256 metadata.", StatPayload, lambda p: self.runtime.stat(p.path, sha256=p.sha256), ActionRisk.SAFE),
            ("files.read_text", "Read bounded UTF-8/text content from a regular text file; never returns raw binary bytes.", ReadPayload, lambda p: self.runtime.read_text(**p.model_dump()), ActionRisk.SAFE),
            ("files.create_directory", "Create a real directory/carpeta and verify it exists.", CreateDirectoryPayload, lambda p: self.runtime.create_directory(**p.model_dump()), ActionRisk.SAFE),
            ("files.create_file", "Create a real UTF-8 file/archivo atomically and verify its content; overwrite is blocked by default.", CreateFilePayload, lambda p: self.runtime.create_file(**p.model_dump()), ActionRisk.SAFE),
            ("files.copy", "Copy a file or directory and verify source and destination; overwrite is blocked by default.", TransferPayload, lambda p: self.runtime.copy(**p.model_dump()), ActionRisk.SENSITIVE),
            ("files.move", "Move a file or directory and verify source removal and destination creation.", TransferPayload, lambda p: self.runtime.move(**p.model_dump()), ActionRisk.SENSITIVE),
            ("files.rename", "Rename a file or directory in its current parent without path traversal.", RenamePayload, lambda p: self.runtime.rename(**p.model_dump()), ActionRisk.SENSITIVE),
            ("files.delete", "Send a disposable file/folder to Recycle Bin, or permanently delete only with explicit security confirmation.", DeletePayload, lambda p: self.runtime.delete(**p.model_dump()), ActionRisk.DESTRUCTIVE),
            ("files.batch_move", "Move up to 100 files with honest per-item partial failure results.", BatchMovePayload, lambda p: self.runtime.batch_move(**p.model_dump()), ActionRisk.SENSITIVE),
            ("files.open", "Open an existing file or folder with its Windows default application.", PathPayload, lambda p: self.runtime.open_with_default_app(p.path), ActionRisk.SENSITIVE),
            ("files.reveal", "Reveal an existing file or folder in Windows Explorer without UI automation.", PathPayload, lambda p: self.runtime.reveal(p.path), ActionRisk.SAFE),
        )
        for name, description, payload, operation, risk in definitions:
            registry.register(ActionDefinition(name=name, description=description, payload_model=payload, handler=self._action_handler(operation), tags=("files", "filesystem", "windows"), risk=risk, idempotent=False, supports_cancellation=name == "files.search"))

    def register_tools(self, registry: ToolRegistry) -> None:
        for definition in registry_actions(self.runtime):
            registry.register(ToolDefinition(name=definition.name, description=definition.description, input_model=definition.payload_model, handler=self._tool_handler(definition.name), tags=("files", "filesystem")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        names = tuple(definition.name for definition in registry_actions(self.runtime))
        registry.register(CapabilityDescriptor(name="files.navigator", module_name=self.name, intent="files", description="Navigate and operate real Windows files and folders with bounded search, verification, and SecurityGate.", action_names=names, tool_names=names, keywords=("file", "files", "archivo", "archivos", "folder", "carpeta", "buscar", "downloads", "descargas", "documents", "documentos", "copy", "move", "rename"), mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION), task_type="assistant"))

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    @staticmethod
    def _action_handler(operation: Callable[[BaseModel], dict[str, Any]]):
        def handler(context: ActionContext, payload: BaseModel) -> ActionResult:
            if context.dry_run:
                return ActionResult(message="file operation validated", data={"operation": context.metadata.get("action")})
            try:
                data = operation(payload)
            except FileOperationError as exc:
                context.event_bus.publish("file_operation_failed", {"run_id": context.metadata.get("agent_run_id") or context.correlation_id, "correlation_id": context.correlation_id, "error_code": exc.code.value})
                return ActionResult(status=ExecutionStatus.FAILED, message=str(exc), data={"error_code": exc.code.value, "path": exc.path})
            operation_name = str(data.get("operation") or "file_operation")
            event_name = {
                "search": "file_search_completed",
                "create_directory": "directory_created",
                "create_file": "file_created",
                "copy": "file_copied",
                "move": "file_moved",
                "rename": "file_renamed",
                "trash": "file_trashed",
                "permanent_delete": "file_deleted",
            }.get(operation_name)
            if event_name:
                context.event_bus.publish(
                    event_name,
                    {
                        "run_id": context.metadata.get("agent_run_id")
                        or context.correlation_id,
                        "correlation_id": context.correlation_id,
                        "verified": bool(data.get("verified", True)),
                    },
                )
            return ActionResult(message="file operation completed and verified", data=data, artifacts=list(data.get("artifacts") or []))

        return handler

    @staticmethod
    def _tool_handler(action_name: str):
        def handler(context: ToolContext, payload: BaseModel) -> ToolResult:
            if context.action_executor is None:
                return ToolResult(status=ToolExecutionStatus.FAILED, message="secured action executor is unavailable", data={"error_code": "SECURITY_BLOCKED"})
            observation = context.action_executor.execute(ActionRequest(run_id=context.correlation_id, call_id=uuid4().hex, action=action_name, arguments=payload.model_dump(mode="json"), correlation_id=context.correlation_id, metadata={**context.metadata, "invoked_via": "tool_registry"}))
            success = observation.status is ActionStatus.SUCCESS
            return ToolResult(status=ToolExecutionStatus.SUCCESS if success else ToolExecutionStatus.FAILED, message=observation.message, data={**observation.data, **({"confirmation_id": observation.confirmation_id} if observation.confirmation_id else {}), **({"error_code": "SECURITY_BLOCKED"} if observation.status is ActionStatus.BLOCKED else {})}, artifacts=observation.artifacts)

        return handler


def registry_actions(runtime: FileNavigator) -> list[ActionDefinition]:
    registry = ActionRegistry()
    FilesModule(runtime).register_actions(registry)
    return registry.list_actions()
