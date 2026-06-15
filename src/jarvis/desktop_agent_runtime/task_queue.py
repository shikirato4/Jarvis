from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTaskQueueItem:
    title: str
    description: str = ""
    task_type: str = "agent"
    status: AgentTaskStatus = AgentTaskStatus.PENDING
    priority: int = 5
    source: str = "desktop_chat"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    next_action: str | None = None
    requires_confirmation: bool = False
    id: str = field(default_factory=lambda: f"task-{uuid4().hex[:10]}")
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.task_type,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "result": self.result,
            "error": self.error,
            "next_action": self.next_action,
            "requires_confirmation": self.requires_confirmation,
        }


class AgentTaskQueue:
    def __init__(self) -> None:
        self._items: list[AgentTaskQueueItem] = []

    def add(
        self,
        title: str,
        *,
        description: str = "",
        task_type: str = "agent",
        priority: int = 5,
        source: str = "desktop_chat",
        requires_confirmation: bool = False,
        next_action: str | None = None,
    ) -> AgentTaskQueueItem:
        item = AgentTaskQueueItem(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            source=source,
            requires_confirmation=requires_confirmation,
            next_action=next_action,
        )
        self._items.append(item)
        self._items.sort(key=lambda entry: (entry.status != AgentTaskStatus.PENDING, entry.priority, entry.created_at))
        return item

    def list(self, *, include_done: bool = False, limit: int = 20) -> list[AgentTaskQueueItem]:
        items = self._items if include_done else [item for item in self._items if item.status in {AgentTaskStatus.PENDING, AgentTaskStatus.RUNNING}]
        return items[:limit]

    def next_pending(self) -> AgentTaskQueueItem | None:
        for item in self._items:
            if item.status == AgentTaskStatus.PENDING:
                return item
        return None

    def cancel(self, item_id: str) -> AgentTaskQueueItem:
        item = self._find(item_id)
        item.status = AgentTaskStatus.CANCELLED
        item.updated_at = _utcnow()
        return item

    def complete(self, item_id: str, *, result: dict[str, Any] | None = None) -> AgentTaskQueueItem:
        item = self._find(item_id)
        item.status = AgentTaskStatus.COMPLETED
        item.result = result or {}
        item.updated_at = _utcnow()
        return item

    def _find(self, item_id: str) -> AgentTaskQueueItem:
        for item in self._items:
            if item.id == item_id:
                return item
        raise KeyError(f"task queue item not found: {item_id}")
