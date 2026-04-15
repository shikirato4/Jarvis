from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.core.safety import ensure_within_roots


class ComposeNotePayload(BaseModel):
    title: str
    objective: str
    findings: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    output_path: str | None = None
    persist_memory: bool = True


class WriterModule:
    name = "writer"
    description = "Structured writing and artifact generation."

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="writer.compose_note",
                description="Create a structured markdown note and optionally persist it.",
                payload_model=ComposeNotePayload,
                handler=self._compose_note,
                rollback=self._rollback_compose_note,
                tags=("writer", "artifact"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="writer.compose",
                module_name=self.name,
                intent="write",
                description="Write a structured note or artifact.",
                action_names=("writer.compose_note",),
                tool_names=("document.compose",),
                keywords=("escribe", "redacta", "write", "draft", "documenta"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                preferred_provider_kinds=("local",),
                task_type="writing",
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _compose_note(self, context: ActionContext, payload: ComposeNotePayload) -> ActionResult:
        content = self._render_markdown(payload)
        output_path = None
        if payload.output_path:
            resolved_path = ensure_within_roots(payload.output_path, (context.settings.resolved_workspace_root,), "writer output")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_path.write_text(content, encoding="utf-8")
            output_path = str(resolved_path)

        memory_id = None
        if payload.persist_memory:
            entry = context.memory.store_memory(
                kind="document",
                content=content,
                source="writer",
                metadata={"title": payload.title, "output_path": output_path},
            )
            memory_id = entry.id

        return ActionResult(
            message="note composed",
            data={
                "title": payload.title,
                "content": content,
                "output_path": output_path,
                "memory_id": memory_id,
            },
            artifacts=[output_path] if output_path else [],
        )

    def _rollback_compose_note(self, context: ActionContext, result: ActionResult) -> None:
        output_path = result.data.get("output_path")
        if output_path:
            path = Path(output_path)
            if path.exists():
                path.unlink()
        memory_id = result.data.get("memory_id")
        if memory_id:
            context.memory.delete_memory(str(memory_id))

    @staticmethod
    def _render_markdown(payload: ComposeNotePayload) -> str:
        lines = [f"# {payload.title}", "", "## Objective", payload.objective, "", "## Findings"]
        if payload.findings:
            lines.extend(f"- {finding}" for finding in payload.findings)
        else:
            lines.append("- No findings were supplied.")
        lines.extend(["", "## References"])
        if payload.references:
            lines.extend(f"- {reference}" for reference in payload.references)
        else:
            lines.append("- No references were supplied.")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("title", "Jarvis note")
        payload.setdefault("objective", request.query or "Generated note")
        return [ActionStep(action="writer.compose_note", payload=payload)]
