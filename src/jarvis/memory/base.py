from __future__ import annotations

from typing import Any, Protocol

from .service import ActivityEntry, AutomationEntry, MemoryEntry


class MemoryBackend(Protocol):
    def create_schema(self) -> None: ...

    def store_memory(
        self,
        *,
        kind: str,
        content: str,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry: ...

    def delete_memory(self, memory_id: str) -> None: ...

    def search_memories(self, query: str, limit: int = 10) -> list[MemoryEntry]: ...

    def list_recent_memories(self, limit: int = 10) -> list[MemoryEntry]: ...

    def record_activity(
        self,
        *,
        correlation_id: str,
        action_name: str,
        status: str,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> ActivityEntry: ...

    def count_activity(self) -> int: ...

    def save_automation(
        self,
        *,
        automation_id: str | None,
        name: str,
        action_name: str,
        payload: dict[str, Any],
        interval_seconds: int,
        enabled: bool,
    ) -> AutomationEntry: ...
