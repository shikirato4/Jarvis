from __future__ import annotations

from typing import Iterable

from jarvis.core.modes import ModeManager

from .base import ModelRequest
from .catalog import ModelCatalog, ModelProfile


class ModelRouter:
    def __init__(self, catalog: ModelCatalog, mode_manager: ModeManager) -> None:
        self._catalog = catalog
        self._mode_manager = mode_manager

    def route(self, request: ModelRequest, *, preferred_provider_order: Iterable[str] = ()) -> list[ModelProfile]:
        candidates = self._catalog.select_for_task(
            logical_name=request.logical_model,
            task_type=request.task_type,
            required_capabilities=request.required_capabilities,
        )
        policy = self._mode_manager.current_policy()
        allowed_kinds = {item.casefold() for item in policy.allowed_provider_kinds}
        if allowed_kinds:
            candidates = [candidate for candidate in candidates if candidate.provider_kind.casefold() in allowed_kinds]
        if not preferred_provider_order:
            return candidates
        ranked: list[ModelProfile] = []
        order = [provider.casefold() for provider in preferred_provider_order]
        for provider_name in order:
            ranked.extend(candidate for candidate in candidates if candidate.provider.casefold() == provider_name and candidate not in ranked)
        ranked.extend(candidate for candidate in candidates if candidate not in ranked)
        return ranked
