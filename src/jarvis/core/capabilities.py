from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Callable

from jarvis.actions.models import ActionStep
from jarvis.core.modes import ExecutionMode
from jarvis.models.base import JarvisBaseModel

if TYPE_CHECKING:
    from jarvis.cognition.models import OrchestrationRequest


PlanBuilder = Callable[["OrchestrationRequest"], list[ActionStep]]


class CapabilityDescriptor(JarvisBaseModel):
    name: str
    module_name: str
    intent: str
    description: str
    action_names: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    mode_policy: tuple[ExecutionMode, ...] = ()
    preferred_provider_kinds: tuple[str, ...] = ()
    task_type: str = "general"
    supports_planning: bool = False


@dataclass(slots=True)
class RegisteredCapability:
    descriptor: CapabilityDescriptor
    plan_builder: PlanBuilder | None = None


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, RegisteredCapability] = {}

    def register(self, descriptor: CapabilityDescriptor, *, plan_builder: PlanBuilder | None = None) -> None:
        self._capabilities[descriptor.intent] = RegisteredCapability(descriptor=descriptor, plan_builder=plan_builder)

    def get(self, intent: str) -> RegisteredCapability | None:
        return self._capabilities.get(intent)

    def list_capabilities(self) -> list[RegisteredCapability]:
        return sorted(self._capabilities.values(), key=lambda item: item.descriptor.intent)

    def infer_intent(self, query: str, *, default_intent: str | None = None) -> str | None:
        lowered = query.casefold()
        tokens = set(re.findall(r"\w+", lowered))
        best_intent: str | None = None
        best_score: tuple[int, int, int] | None = None
        for index, capability in enumerate(self.list_capabilities()):
            matched_keywords = [
                keyword.casefold()
                for keyword in capability.descriptor.keywords
                if self._matches_keyword(keyword.casefold(), lowered, tokens)
            ]
            if not matched_keywords:
                continue
            score = (
                len(matched_keywords),
                max(len(keyword) for keyword in matched_keywords),
                -index,
            )
            if best_score is None or score > best_score:
                best_intent = capability.descriptor.intent
                best_score = score
        if best_intent is not None:
            return best_intent
        return default_intent

    def build_plan(self, intent: str, request: "OrchestrationRequest") -> list[ActionStep]:
        capability = self.get(intent)
        if capability is None:
            raise KeyError(intent)
        if capability.plan_builder is not None:
            return capability.plan_builder(request)
        if not capability.descriptor.action_names:
            raise ValueError(f"capability '{intent}' does not expose actions")
        payload = dict(request.payload)
        if request.query and "query" not in payload:
            payload["query"] = request.query
        return [ActionStep(action=capability.descriptor.action_names[0], payload=payload)]

    @staticmethod
    def _matches_keyword(keyword: str, lowered_query: str, tokens: set[str]) -> bool:
        if " " in keyword:
            return keyword in lowered_query
        return keyword in tokens
