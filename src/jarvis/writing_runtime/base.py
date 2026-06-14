from __future__ import annotations

import json
from typing import Any

from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.service import ModelService


class WritingModelAdapter:
    def __init__(self, models: ModelService) -> None:
        self._models = models

    def infer_json(
        self,
        *,
        task_type: str,
        logical_model: str | None,
        prompt: str,
        correlation_id: str,
        metadata: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        settings = getattr(self._models, "_settings", None)
        if settings is not None and not getattr(settings, "ollama_enabled", True):
            return None
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
                    timeout_seconds=timeout_seconds,
                    metadata=metadata,
                )
            )
            parsed = json.loads(response.content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
