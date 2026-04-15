from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .models import ActivityRecord, AutomationRecord, MemoryRecord
from .repository import MemoryRepository


def _loads(raw: str) -> dict[str, Any]:
    return json.loads(raw or "{}")


class MemoryEntry(JarvisBaseModel):
    id: str
    kind: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ActivityEntry(JarvisBaseModel):
    id: str
    correlation_id: str
    action_name: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AutomationEntry(JarvisBaseModel):
    id: str
    name: str
    action_name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int
    enabled: bool
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None


class MemoryService:
    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    def create_schema(self) -> None:
        self._repository.create_schema()

    def store_memory(
        self,
        *,
        kind: str,
        content: str,
        source: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        record = self._repository.add_memory(kind=kind, content=content, source=source, metadata=metadata or {})
        return self._to_memory_entry(record)

    def delete_memory(self, memory_id: str) -> None:
        self._repository.delete_memory(memory_id)

    def search_memories(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        return [self._to_memory_entry(record) for record in self._repository.search_memories(query, limit)]

    def list_recent_memories(self, limit: int = 10) -> list[MemoryEntry]:
        return [self._to_memory_entry(record) for record in self._repository.list_memories(limit)]

    def record_activity(
        self,
        *,
        correlation_id: str,
        action_name: str,
        status: str,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> ActivityEntry:
        record = self._repository.record_activity(
            correlation_id=correlation_id,
            action_name=action_name,
            status=status,
            payload=payload,
            result=result,
        )
        return self._to_activity_entry(record)

    def count_activity(self) -> int:
        return self._repository.count_activity()

    def save_automation(
        self,
        *,
        automation_id: str | None,
        name: str,
        action_name: str,
        payload: dict[str, Any],
        interval_seconds: int,
        enabled: bool,
    ) -> AutomationEntry:
        record = self._repository.upsert_automation(
            automation_id=automation_id,
            name=name,
            action_name=action_name,
            payload=payload,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )
        return self._to_automation_entry(record)

    def get_automation(self, automation_id: str) -> AutomationEntry | None:
        record = self._repository.get_automation(automation_id)
        return self._to_automation_entry(record) if record else None

    def list_automations(self, *, enabled_only: bool = False) -> list[AutomationEntry]:
        return [self._to_automation_entry(record) for record in self._repository.list_automations(enabled_only=enabled_only)]

    def update_automation_runtime(
        self,
        automation_id: str,
        *,
        last_run_at: datetime | None,
        next_run_at: datetime | None,
    ) -> None:
        self._repository.update_automation_runtime(
            automation_id,
            last_run_at=last_run_at,
            next_run_at=next_run_at,
        )

    @staticmethod
    def _to_memory_entry(record: MemoryRecord) -> MemoryEntry:
        return MemoryEntry(
            id=record.id,
            kind=record.kind,
            content=record.content,
            source=record.source,
            metadata=_loads(record.metadata_json),
            created_at=record.created_at,
        )

    @staticmethod
    def _to_activity_entry(record: ActivityRecord) -> ActivityEntry:
        return ActivityEntry(
            id=record.id,
            correlation_id=record.correlation_id,
            action_name=record.action_name,
            status=record.status,
            payload=_loads(record.payload_json),
            result=_loads(record.result_json),
            created_at=record.created_at,
        )

    @staticmethod
    def _to_automation_entry(record: AutomationRecord) -> AutomationEntry:
        return AutomationEntry(
            id=record.id,
            name=record.name,
            action_name=record.action_name,
            payload=_loads(record.payload_json),
            interval_seconds=record.interval_seconds,
            enabled=record.enabled,
            last_run_at=record.last_run_at,
            next_run_at=record.next_run_at,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
