from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult, ActionStep
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode
from jarvis.writing_runtime.models import WritingContinuationRequest, WritingMode


class WritingContinuePayload(BaseModel):
    prompt: str
    instruction: str | None = None
    mode: WritingMode = WritingMode.COPILOT
    target_window: str | None = None
    ensure_window_contains: str | None = None
    desired_words: int = 120
    preserve_style: bool = True
    preserve_narrative_state: bool = True
    write_directly: bool = True
    collection_name: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class WritingLookupPayload(BaseModel):
    task_id: str


class WritingModule:
    name = "writing"
    description = "Live writing copilot runtime and autonomous continuation."

    def __init__(self, writing_runtime) -> None:
        self._writing = writing_runtime

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="writing.continue_task",
                description="Continue writing in the active document while preserving context and style.",
                payload_model=WritingContinuePayload,
                handler=self._continue,
                tags=("writing", "copilot"),
            )
        )
        registry.register(
            ActionDefinition(
                name="writing.analyze_context",
                description="Analyze the current writing context and style.",
                payload_model=WritingContinuePayload,
                handler=self._analyze,
                tags=("writing", "analysis"),
            )
        )

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="writing.copilot",
                module_name=self.name,
                intent="writing",
                description="Continue the current document while preserving user style and coherence.",
                action_names=("writing.continue_task", "writing.analyze_context"),
                keywords=("continúa mi libro", "sigue donde me quedé", "continúa con el mismo tono", "escribe como yo", "continúa la escena", "writing"),
                mode_policy=(ExecutionMode.ASSIST, ExecutionMode.RESEARCH, ExecutionMode.OPERATOR, ExecutionMode.AUTOMATION),
                task_type="writing",
                supports_planning=True,
            ),
            plan_builder=self._build_plan,
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _continue(self, context: ActionContext, payload: WritingContinuePayload) -> ActionResult:
        receipt = self._writing.continue_writing(
            WritingContinuationRequest(
                prompt=payload.prompt,
                instruction=payload.instruction,
                mode=payload.mode,
                target_window=payload.target_window,
                ensure_window_contains=payload.ensure_window_contains,
                desired_words=payload.desired_words,
                preserve_style=payload.preserve_style,
                preserve_narrative_state=payload.preserve_narrative_state,
                write_directly=payload.write_directly,
                collection_name=payload.collection_name,
                metadata={**payload.metadata, **context.metadata},
            )
        )
        return ActionResult(message=receipt.message, data=receipt.model_dump(mode="json"))

    def _analyze(self, context: ActionContext, payload: WritingContinuePayload) -> ActionResult:
        analysis = self._writing.analyze(
            WritingContinuationRequest(
                prompt=payload.prompt,
                instruction=payload.instruction,
                mode=payload.mode,
                target_window=payload.target_window,
                ensure_window_contains=payload.ensure_window_contains,
                desired_words=payload.desired_words,
                preserve_style=payload.preserve_style,
                preserve_narrative_state=payload.preserve_narrative_state,
                write_directly=payload.write_directly,
                collection_name=payload.collection_name,
                metadata={**payload.metadata, **context.metadata},
            )
        )
        return ActionResult(message="writing analysis ready", data=analysis.model_dump(mode="json"))

    @staticmethod
    def _build_plan(request) -> list[ActionStep]:
        payload = dict(request.payload)
        payload.setdefault("prompt", request.query or "")
        return [ActionStep(action="writing.continue_task", payload=payload)]
