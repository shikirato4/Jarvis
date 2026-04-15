from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select

from jarvis.memory.models import ResearchTaskEventRecord, ResearchTaskRecord
from jarvis.memory.repository import Database

from .models import ResearchTask


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _loads(raw: str | None) -> dict[str, Any]:
    return json.loads(raw or "{}")


class ResearchRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def upsert_task(self, task: ResearchTask) -> ResearchTaskRecord:
        with self._database.session_scope() as session:
            record = session.get(ResearchTaskRecord, task.task_id)
            payload = task.model_dump(mode="json")
            report = task.report.model_dump(mode="json") if task.report else {}
            if record is None:
                record = ResearchTaskRecord(
                    id=task.task_id,
                    query=task.query,
                    status=task.status.value,
                    mission_id=task.mission_id,
                    snapshot_json=_dumps(payload),
                    report_json=_dumps(report),
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                session.add(record)
            else:
                record.query = task.query
                record.status = task.status.value
                record.mission_id = task.mission_id
                record.snapshot_json = _dumps(payload)
                record.report_json = _dumps(report)
                record.updated_at = task.updated_at
            session.flush()
            session.refresh(record)
            return record

    def get_task(self, task_id: str) -> ResearchTask | None:
        with self._database.session_scope() as session:
            record = session.get(ResearchTaskRecord, task_id)
            if record is None:
                return None
            return ResearchTask.model_validate(_loads(record.snapshot_json))

    def list_tasks(self, limit: int = 20) -> list[ResearchTask]:
        with self._database.session_scope() as session:
            statement = select(ResearchTaskRecord).order_by(desc(ResearchTaskRecord.updated_at)).limit(limit)
            return [ResearchTask.model_validate(_loads(record.snapshot_json)) for record in session.scalars(statement)]

    def append_event(self, *, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._database.session_scope() as session:
            session.add(ResearchTaskEventRecord(task_id=task_id, event_type=event_type, payload_json=_dumps(payload)))
