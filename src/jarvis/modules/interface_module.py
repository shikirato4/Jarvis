from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep, ExecutionStatus
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.memory_semantic.service import SemanticMemoryService
from jarvis.tools.models import ToolExecutionStatus, ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry
from jarvis.ui_automation.base import (
    CancellationRequest,
    ClickRequest,
    FocusWindowRequest,
    InsertBlocksRequest,
    MoveMouseRequest,
    ShortcutRequest,
    UIAutomationMode,
    WriteTextRequest,
)
from jarvis.ui_automation.service import UIAutomationService


class ActiveWindowPayload(BaseModel):
    pass


class ContextualDirectWritePayload(BaseModel):
    prompt: str
    collection_name: str | None = None
    top_k: int | None = None
    target_window: str | None = None
    mode: UIAutomationMode = UIAutomationMode.COPILOT


class InterfaceModule:
    name = "interface"
    description = "Desktop interface automation, window control and direct writing."

    def __init__(self, ui_service: UIAutomationService, semantic_memory: SemanticMemoryService) -> None:
        self._ui = ui_service
        self._semantic = semantic_memory

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(ActionDefinition(name="interface.active_window", description="Inspect the active desktop window.", payload_model=ActiveWindowPayload, handler=self._active_window, tags=("interface", "inspection")))
        registry.register(ActionDefinition(name="interface.focus_window", description="Focus a desktop window by title or handle.", payload_model=FocusWindowRequest, handler=self._focus_window, tags=("interface", "window")))
        registry.register(ActionDefinition(name="interface.write_text", description="Write text into the active window.", payload_model=WriteTextRequest, handler=self._write_text, tags=("interface", "typing")))
        registry.register(ActionDefinition(name="interface.insert_blocks", description="Insert text by blocks into the active window.", payload_model=InsertBlocksRequest, handler=self._insert_blocks, tags=("interface", "typing")))
        registry.register(ActionDefinition(name="interface.move_mouse", description="Move the mouse pointer.", payload_model=MoveMouseRequest, handler=self._move_mouse, tags=("interface", "mouse")))
        registry.register(ActionDefinition(name="interface.click_mouse", description="Perform a mouse click.", payload_model=ClickRequest, handler=self._click_mouse, tags=("interface", "mouse")))
        registry.register(ActionDefinition(name="interface.keyboard_shortcut", description="Send a keyboard shortcut.", payload_model=ShortcutRequest, handler=self._keyboard_shortcut, tags=("interface", "keyboard")))
        registry.register(ActionDefinition(name="interface.cancel", description="Cancel an in-flight UI operation.", payload_model=CancellationRequest, handler=self._cancel_operation, tags=("interface", "control")))
        registry.register(ActionDefinition(name="interface.contextual_direct_write", description="Recover semantic context and write directly into the active editor.", payload_model=ContextualDirectWritePayload, handler=self._contextual_direct_write, tags=("interface", "typing", "semantic")))

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(ToolDefinition(name="desktop.active_window", description="Get the active desktop window.", input_model=ActiveWindowPayload, handler=self._tool_active_window, tags=("interface", "state")))
        registry.register(ToolDefinition(name="desktop.write_text", description="Write text into the active desktop window.", input_model=WriteTextRequest, handler=self._tool_write_text, tags=("interface", "typing")))
        registry.register(ToolDefinition(name="desktop.shortcut", description="Send a keyboard shortcut to the active window.", input_model=ShortcutRequest, handler=self._tool_keyboard_shortcut, tags=("interface", "keyboard")))

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="interface.desktop_control",
                module_name=self.name,
                intent="ui_control",
                description="Control windows, mouse, keyboard and direct editor interaction.",
                action_names=("interface.active_window", "interface.focus_window", "interface.write_text", "interface.move_mouse", "interface.click_mouse", "interface.keyboard_shortcut"),
                tool_names=("desktop.active_window", "desktop.write_text", "desktop.shortcut"),
                keywords=("ventana activa", "window", "mouse", "teclado", "keyboard", "write in word"),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="assistant",
            )
        )
        registry.register(
            CapabilityDescriptor(
                name="interface.direct_write",
                module_name=self.name,
                intent="direct_write",
                description="Write directly into the active editor or application.",
                action_names=("interface.write_text", "interface.insert_blocks", "interface.contextual_direct_write"),
                tool_names=("desktop.write_text",),
                keywords=("escribe directo", "write directly", "word", "editor activo", "copiloto"),
                mode_policy=(ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="writing",
                supports_planning=False,
            ),
            plan_builder=self._build_direct_write_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _active_window(self, context: ActionContext, payload: ActiveWindowPayload) -> ActionResult:
        receipt = self._ui.active_window(correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _focus_window(self, context: ActionContext, payload: FocusWindowRequest) -> ActionResult:
        receipt = self._ui.focus_window(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _write_text(self, context: ActionContext, payload: WriteTextRequest) -> ActionResult:
        receipt = self._ui.write_text(payload, correlation_id=context.correlation_id)
        return self._action_result_from_ui_receipt(receipt)

    def _insert_blocks(self, context: ActionContext, payload: InsertBlocksRequest) -> ActionResult:
        receipt = self._ui.insert_blocks(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _move_mouse(self, context: ActionContext, payload: MoveMouseRequest) -> ActionResult:
        receipt = self._ui.move_mouse(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _click_mouse(self, context: ActionContext, payload: ClickRequest) -> ActionResult:
        receipt = self._ui.click(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _keyboard_shortcut(self, context: ActionContext, payload: ShortcutRequest) -> ActionResult:
        receipt = self._ui.hotkey(payload, correlation_id=context.correlation_id)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _cancel_operation(self, context: ActionContext, payload: CancellationRequest) -> ActionResult:
        receipt = self._ui.cancel(payload)
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _contextual_direct_write(self, context: ActionContext, payload: ContextualDirectWritePayload) -> ActionResult:
        retrieved = self._semantic.retrieve_context(
            SemanticSearchQuery(
                query=payload.prompt,
                collection_name=payload.collection_name,
                top_k=payload.top_k,
                correlation_id=context.correlation_id,
            )
        )
        lines = [payload.prompt, ""]
        if retrieved.summary:
            lines.extend(["Context:", retrieved.summary, ""])
        if retrieved.sources:
            lines.extend(["Sources:"])
            lines.extend(f"- {item}" for item in retrieved.sources)
        receipt = self._ui.write_text(
            WriteTextRequest(
                text="\n".join(lines).strip(),
                mode=payload.mode,
                focus_target=payload.target_window,
            ),
            correlation_id=context.correlation_id,
        )
        result = self._action_result_from_ui_receipt(receipt)
        result.data["retrieved_context"] = retrieved.model_dump(mode="json")
        return result

    def _tool_active_window(self, context: ToolContext, payload: ActiveWindowPayload) -> ToolResult:
        receipt = self._ui.active_window(correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _tool_write_text(self, context: ToolContext, payload: WriteTextRequest) -> ToolResult:
        receipt = self._ui.write_text(payload, correlation_id=context.correlation_id)
        return ToolResult(
            status=ToolExecutionStatus.SUCCESS if receipt.success else ToolExecutionStatus.FAILED,
            message=receipt.message,
            data=receipt.model_dump(mode="json"),
        )

    def _tool_keyboard_shortcut(self, context: ToolContext, payload: ShortcutRequest) -> ToolResult:
        receipt = self._ui.hotkey(payload, correlation_id=context.correlation_id)
        return ToolResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    @staticmethod
    def _build_direct_write_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("text", request.query or "")
        payload.setdefault("mode", UIAutomationMode.COPILOT.value)
        return [ActionStep(action="interface.write_text", payload=payload)]

    @staticmethod
    def _action_result_from_ui_receipt(receipt) -> ActionResult:
        return ActionResult(
            status=ExecutionStatus.SUCCESS if receipt.success else ExecutionStatus.FAILED,
            message=receipt.message,
            data=receipt.model_dump(mode="json"),
        )
