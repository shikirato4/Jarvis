from __future__ import annotations

from typing import Any
from uuid import uuid4

from jarvis.actions.router import ActionRouter
from jarvis.cognition.models import OrchestrationRequest
from jarvis.cognition.orchestrator import CognitiveOrchestrator
from jarvis.core.capabilities import CapabilityRegistry
from jarvis.core.errors import JarvisError, TaskRoutingError
from jarvis.core.metacommands import MetaCommandKind, MetaCommandParser
from jarvis.core.modes import ModeManager
from jarvis.core.state import RuntimeStateManager
from jarvis.tools.registry import ToolRegistry

from .models import RouteType, TaskRequest, TaskResponse


class TaskRouter:
    def __init__(
        self,
        *,
        mode_manager: ModeManager,
        state_manager: RuntimeStateManager,
        meta_command_parser: MetaCommandParser,
        capability_registry: CapabilityRegistry,
        action_router: ActionRouter,
        tool_registry: ToolRegistry,
        orchestrator: CognitiveOrchestrator,
    ) -> None:
        self._mode_manager = mode_manager
        self._state_manager = state_manager
        self._meta_command_parser = meta_command_parser
        self._capability_registry = capability_registry
        self._action_router = action_router
        self._tool_registry = tool_registry
        self._orchestrator = orchestrator

    def route(self, request: TaskRequest) -> TaskResponse:
        task_id = str(uuid4())
        target = request.intent or "runtime"
        route_type = RouteType.METACOMMAND if request.raw_input and request.raw_input.strip().startswith("/") else RouteType.ORCHESTRATION
        source = "text" if request.raw_input else "structured"
        self._state_manager.begin_task(
            task_id=task_id,
            route_type=route_type.value,
            target=target,
            source=source,
            metadata=request.metadata,
        )
        try:
            if request.raw_input and request.raw_input.strip().startswith("/"):
                response = self._handle_metacommand(task_id, request)
            else:
                response = self._handle_freeform_task(task_id, request)
            self._state_manager.complete_task(task_id, output_summary=response.message)
            return response
        except JarvisError as exc:
            self._state_manager.fail_task(task_id, error=exc.to_dict())
            raise
        except Exception as exc:
            wrapped = TaskRoutingError(str(exc))
            self._state_manager.fail_task(task_id, error=wrapped.to_dict())
            raise wrapped from exc

    def _handle_metacommand(self, task_id: str, request: TaskRequest) -> TaskResponse:
        assert request.raw_input is not None
        command = self._meta_command_parser.parse(request.raw_input)

        if command.kind == MetaCommandKind.HELP:
            snapshot = self._snapshot()
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.METACOMMAND,
                target="help",
                mode=self._mode_manager.current_mode().value,
                message="metacommand help generated",
                meta_command=command,
                state_snapshot=snapshot,
            )

        if command.kind == MetaCommandKind.STATE:
            snapshot = self._snapshot()
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.STATE,
                target="runtime.snapshot",
                mode=self._mode_manager.current_mode().value,
                message="runtime snapshot generated",
                meta_command=command,
                state_snapshot=snapshot,
            )

        if command.kind == MetaCommandKind.MODE:
            self._mode_manager.set_mode(
                command.target or self._mode_manager.current_mode().value,
                reason=str(command.payload.get("reason") or "metacommand"),
                sticky=bool(command.payload.get("sticky", True)),
            )
            snapshot = self._snapshot()
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.METACOMMAND,
                target=command.target or "mode",
                mode=self._mode_manager.current_mode().value,
                message=f"mode changed to {self._mode_manager.current_mode().value}",
                meta_command=command,
                state_snapshot=snapshot,
            )

        if command.kind == MetaCommandKind.ACTION:
            if command.target is None:
                raise TaskRoutingError("action metacommand requires a target action")
            self._mode_manager.validate_action(command.target)
            receipt = self._action_router.execute(
                command.target,
                command.payload,
                correlation_id=task_id,
                dry_run=request.dry_run,
                metadata=request.metadata,
            )
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.ACTION,
                target=command.target,
                mode=self._mode_manager.current_mode().value,
                message=receipt.message,
                meta_command=command,
                action_receipt=receipt,
            )

        if command.kind == MetaCommandKind.TOOL:
            if command.target is None:
                raise TaskRoutingError("tool metacommand requires a target tool")
            definition = self._tool_registry.get(command.target)
            if definition is None:
                raise TaskRoutingError(f"tool '{command.target}' is not registered")
            self._mode_manager.validate_tool_tags(command.target, definition.tags)
            receipt = self._tool_registry.invoke(
                command.target,
                command.payload,
                correlation_id=task_id,
                dry_run=request.dry_run,
                metadata=request.metadata,
            )
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.TOOL,
                target=command.target,
                mode=self._mode_manager.current_mode().value,
                message=receipt.message,
                meta_command=command,
                tool_receipt=receipt,
            )

        if command.kind == MetaCommandKind.TASK:
            task_request = TaskRequest(
                intent=command.target,
                payload=command.payload,
                metadata=request.metadata,
                dry_run=request.dry_run,
            )
            return self._handle_freeform_task(task_id, task_request, meta_command=command)

        if command.kind == MetaCommandKind.REMEMBER:
            self._mode_manager.validate_action("memory.store")
            receipt = self._action_router.execute(
                "memory.store",
                command.payload,
                correlation_id=task_id,
                dry_run=request.dry_run,
                metadata=request.metadata,
            )
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.ACTION,
                target="memory.store",
                mode=self._mode_manager.current_mode().value,
                message=receipt.message,
                meta_command=command,
                action_receipt=receipt,
            )

        if command.kind == MetaCommandKind.RECALL:
            self._mode_manager.validate_action("memory.search")
            receipt = self._action_router.execute(
                "memory.search",
                command.payload,
                correlation_id=task_id,
                dry_run=request.dry_run,
                metadata=request.metadata,
            )
            return TaskResponse(
                task_id=task_id,
                route_type=RouteType.ACTION,
                target="memory.search",
                mode=self._mode_manager.current_mode().value,
                message=receipt.message,
                meta_command=command,
                action_receipt=receipt,
            )

        raise TaskRoutingError(f"unsupported metacommand '{command.kind.value}'")

    def _handle_freeform_task(
        self,
        task_id: str,
        request: TaskRequest,
        *,
        meta_command=None,
    ) -> TaskResponse:
        resolved_intent = self._mode_manager.resolve_intent(request.intent)
        self._validate_intent(resolved_intent)
        orchestration_request = OrchestrationRequest(
            intent=resolved_intent,
            query=request.raw_input,
            payload=request.payload,
            persist_input=bool(request.raw_input),
        )
        response = self._orchestrator.handle(orchestration_request)
        return TaskResponse(
            task_id=task_id,
            route_type=RouteType.ORCHESTRATION,
            target=response.resolved_intent,
            mode=self._mode_manager.current_mode().value,
            message=f"orchestrated {response.resolved_intent}",
            meta_command=meta_command,
            orchestration=response,
        )

    def _validate_intent(self, intent: str | None) -> None:
        if intent is None:
            return
        capability = self._capability_registry.get(intent)
        if capability is None:
            raise TaskRoutingError(f"intent '{intent}' is not registered")
        if capability.descriptor.mode_policy and self._mode_manager.current_mode() not in capability.descriptor.mode_policy:
            raise TaskRoutingError(
                f"intent '{intent}' is not allowed while mode is '{self._mode_manager.current_mode().value}'"
            )
        for action_name in capability.descriptor.action_names:
            self._mode_manager.validate_action(action_name)

    def _snapshot(self):
        return self._state_manager.snapshot(
            action_names=[definition.name for definition in self._action_router.registry.list_actions()],
            tool_names=[definition.name for definition in self._tool_registry.list_tools()],
        )
