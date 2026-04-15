from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import desc, func, or_, select
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from jarvis.core.errors import PersistenceError

from .models import ActivityRecord, AutomationRecord, Base, MemoryRecord


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


class Database:
    def __init__(self, database_url: str, engine_options: dict[str, Any] | None = None) -> None:
        self.engine = create_engine(database_url, future=True, **(engine_options or {}))
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, class_=Session)

    def create_schema(self) -> None:
        try:
            Base.metadata.create_all(self.engine)
        except OperationalError as exc:
            if "already exists" in str(exc).lower():
                return
            raise PersistenceError(str(exc)) from exc

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            raise PersistenceError(str(exc)) from exc
        finally:
            session.close()


class MemoryRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def add_memory(self, kind: str, content: str, source: str, metadata: dict[str, Any]) -> MemoryRecord:
        with self._database.session_scope() as session:
            record = MemoryRecord(kind=kind, content=content, source=source, metadata_json=_dumps(metadata))
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def delete_memory(self, memory_id: str) -> None:
        with self._database.session_scope() as session:
            record = session.get(MemoryRecord, memory_id)
            if record is not None:
                session.delete(record)

    def search_memories(self, query: str, limit: int) -> list[MemoryRecord]:
        pattern = f"%{query}%"
        with self._database.session_scope() as session:
            statement = (
                select(MemoryRecord)
                .where(or_(MemoryRecord.content.ilike(pattern), MemoryRecord.metadata_json.ilike(pattern)))
                .order_by(desc(MemoryRecord.created_at))
                .limit(limit)
            )
            return list(session.scalars(statement))

    def list_memories(self, limit: int) -> list[MemoryRecord]:
        with self._database.session_scope() as session:
            statement = select(MemoryRecord).order_by(desc(MemoryRecord.created_at)).limit(limit)
            return list(session.scalars(statement))

    def record_activity(
        self,
        correlation_id: str,
        action_name: str,
        status: str,
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> ActivityRecord:
        with self._database.session_scope() as session:
            record = ActivityRecord(
                correlation_id=correlation_id,
                action_name=action_name,
                status=status,
                payload_json=_dumps(payload),
                result_json=_dumps(result),
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def count_activity(self) -> int:
        with self._database.session_scope() as session:
            return int(session.scalar(select(func.count()).select_from(ActivityRecord)) or 0)

    def upsert_automation(
        self,
        *,
        automation_id: str | None,
        name: str,
        action_name: str,
        payload: dict[str, Any],
        interval_seconds: int,
        enabled: bool,
    ) -> AutomationRecord:
        with self._database.session_scope() as session:
            record = session.get(AutomationRecord, automation_id) if automation_id else None
            if record is None:
                record = AutomationRecord(
                    name=name,
                    action_name=action_name,
                    payload_json=_dumps(payload),
                    interval_seconds=interval_seconds,
                    enabled=enabled,
                )
                session.add(record)
            else:
                record.name = name
                record.action_name = action_name
                record.payload_json = _dumps(payload)
                record.interval_seconds = interval_seconds
                record.enabled = enabled
            session.flush()
            session.refresh(record)
            return record

    def get_automation(self, automation_id: str) -> AutomationRecord | None:
        with self._database.session_scope() as session:
            return session.get(AutomationRecord, automation_id)

    def list_automations(self, *, enabled_only: bool = False) -> list[AutomationRecord]:
        with self._database.session_scope() as session:
            statement = select(AutomationRecord).order_by(AutomationRecord.name.asc())
            if enabled_only:
                statement = statement.where(AutomationRecord.enabled.is_(True))
            return list(session.scalars(statement))

    def update_automation_runtime(
        self,
        automation_id: str,
        *,
        last_run_at: datetime | None,
        next_run_at: datetime | None,
    ) -> None:
        with self._database.session_scope() as session:
            record = session.get(AutomationRecord, automation_id)
            if record is None:
                return
            record.last_run_at = last_run_at
            record.next_run_at = next_run_at
