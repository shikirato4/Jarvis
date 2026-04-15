from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select

from jarvis.memory.models import WritingTaskRecord
from jarvis.memory.repository import Database

from .models import WritingTask


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _loads(raw: str | None) -> dict[str, Any]:
    return json.loads(raw or "{}")


class WritingRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def upsert_task(self, task: WritingTask) -> WritingTaskRecord:
        with self._database.session_scope() as session:
            record = session.get(WritingTaskRecord, task.task_id)
            payload = task.model_dump(mode="json")
            if record is None:
                record = WritingTaskRecord(
                    id=task.task_id,
                    goal=task.goal,
                    status=task.status.value,
                    mission_id=task.mission_id,
                    snapshot_json=_dumps(payload),
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                session.add(record)
            else:
                record.goal = task.goal
                record.status = task.status.value
                record.mission_id = task.mission_id
                record.snapshot_json = _dumps(payload)
                record.updated_at = task.updated_at
            session.flush()
            session.refresh(record)
            return record

    def get_task(self, task_id: str) -> WritingTask | None:
        with self._database.session_scope() as session:
            record = session.get(WritingTaskRecord, task_id)
            if record is None:
                return None
            return WritingTask.model_validate(_loads(record.snapshot_json))

    def list_tasks(self, limit: int = 20) -> list[WritingTask]:
        with self._database.session_scope() as session:
            statement = select(WritingTaskRecord).order_by(desc(WritingTaskRecord.updated_at)).limit(limit)
            return [WritingTask.model_validate(_loads(record.snapshot_json)) for record in session.scalars(statement)]
