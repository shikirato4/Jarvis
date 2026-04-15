from __future__ import annotations

from jarvis.actions.models import ActionStep
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode


def install_cognitive_capabilities(registry: CapabilityRegistry) -> None:
    registry.register(
        CapabilityDescriptor(
            name="cognition.research_brief",
            module_name="cognition",
            intent="research_brief",
            description="Research a topic in the workspace and synthesize a written brief.",
            action_names=("research.workspace_search", "writer.compose_note"),
            tool_names=("workspace.search", "document.compose"),
            keywords=("brief", "informe", "research brief", "resumen investigacion"),
            mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
            preferred_provider_kinds=("local",),
            task_type="planning",
            supports_planning=True,
        ),
        plan_builder=_build_research_brief_plan,
    )
    registry.register(
        CapabilityDescriptor(
            name="cognition.semantic_search",
            module_name="cognition",
            intent="semantic_search",
            description="Recover semantically relevant context from indexed corpora.",
            action_names=(),
            tool_names=(),
            keywords=("contexto", "semantic", "semantico", "recupera contexto", "busca semantica"),
            mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
            preferred_provider_kinds=("local",),
            task_type="reasoning",
            supports_planning=False,
        )
    )
    registry.register(
        CapabilityDescriptor(
            name="cognition.document_ingest",
            module_name="cognition",
            intent="document_ingest",
            description="Ingest a document into semantic memory.",
            action_names=(),
            tool_names=(),
            keywords=("indexa documento", "ingesta documento", "ingest", "index document"),
            mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
            preferred_provider_kinds=("local",),
            task_type="assistant",
            supports_planning=False,
        )
    )
    registry.register(
        CapabilityDescriptor(
            name="cognition.contextual_writing",
            module_name="cognition",
            intent="contextual_writing",
            description="Write with retrieved context from semantic memory.",
            action_names=("writer.compose_note",),
            tool_names=("document.compose",),
            keywords=("escritura contextual", "write with context", "usa contexto", "continua borrador"),
            mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
            preferred_provider_kinds=("local",),
            task_type="writing",
            supports_planning=True,
        )
    )


def _build_research_brief_plan(request) -> list[ActionStep]:
    research_payload = {
        "query": request.payload.get("query", request.query or ""),
        "limit": request.payload.get("limit", 10),
    }
    writer_payload = {
        "title": request.payload.get("title", "Research brief"),
        "objective": request.payload.get("objective", request.query or "Workspace research brief"),
        "output_path": request.payload.get("output_path"),
        "persist_memory": request.payload.get("persist_memory", True),
    }
    return [
        ActionStep(action="research.workspace_search", payload=research_payload),
        ActionStep(action="writer.compose_note", payload=writer_payload),
    ]
