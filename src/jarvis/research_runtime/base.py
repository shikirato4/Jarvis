from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Protocol
from uuid import uuid4

from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.service import ModelService


@dataclass(slots=True)
class ResearchExecutionContext:
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchModelAdapter(Protocol):
    def infer_json(self, *, task_type: str, logical_model: str | None, prompt: str, correlation_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None: ...


class ModelServiceResearchAdapter:
    def __init__(self, models: ModelService) -> None:
        self._models = models

    def infer_json(self, *, task_type: str, logical_model: str | None, prompt: str, correlation_id: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = self._models.infer(
                ModelRequest(
                    task_type=task_type,
                    logical_model=logical_model,
                    required_capabilities=(task_type,),
                    correlation_id=correlation_id,
                    messages=[
                        ModelMessage(role="system", content="Return JSON only."),
                        ModelMessage(role="user", content=prompt),
                    ],
                    metadata=metadata,
                )
            )
            payload = json.loads(response.content)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
