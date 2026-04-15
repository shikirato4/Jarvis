from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select

from jarvis.memory.models import MissionApprovalRecordORM, MissionEventRecord, MissionRecord
from jarvis.memory.repository import Database


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _loads(raw: str | None) -> dict[str, Any]:
    return json.loads(raw or "{}")


class MissionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def upsert_mission(
        self,
        *,
        mission_id: str,
        status: str,
        goal: str,
        active_step_id: str | None,
        requires_human_attention: bool,
        snapshot: dict[str, Any],
        created_at,
        updated_at,
    ) -> MissionRecord:
        with self._database.session_scope() as session:
            record = session.get(MissionRecord, mission_id)
            if record is None:
                record = MissionRecord(
                    id=mission_id,
                    status=status,
                    goal=goal,
                    active_step_id=active_step_id,
                    requires_human_attention=requires_human_attention,
                    snapshot_json=_dumps(snapshot),
                    created_at=created_at,
                    updated_at=updated_at,
                )
                session.add(record)
            else:
                record.status = status
                record.goal = goal
                record.active_step_id = active_step_id
                record.requires_human_attention = requires_human_attention
                record.snapshot_json = _dumps(snapshot)
                record.updated_at = updated_at
            session.flush()
            session.refresh(record)
            return record

    def get_mission(self, mission_id: str) -> MissionRecord | None:
        with self._database.session_scope() as session:
            return session.get(MissionRecord, mission_id)

    def list_missions(self) -> list[MissionRecord]:
        with self._database.session_scope() as session:
            statement = select(MissionRecord).order_by(desc(MissionRecord.created_at))
            return list(session.scalars(statement))

    def list_active_missions(self) -> list[MissionRecord]:
        with self._database.session_scope() as session:
            statement = (
                select(MissionRecord)
                .where(MissionRecord.status.not_in(("completed", "failed", "cancelled", "stopped")))
                .order_by(desc(MissionRecord.updated_at))
            )
            return list(session.scalars(statement))

    def append_event(
        self,
        *,
        mission_id: str,
        event_type: str,
        step_id: str | None,
        payload: dict[str, Any],
    ) -> MissionEventRecord:
        with self._database.session_scope() as session:
            record = MissionEventRecord(
                mission_id=mission_id,
                event_type=event_type,
                step_id=step_id,
                payload_json=_dumps(payload),
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def list_events(self, mission_id: str, *, limit: int | None = None) -> list[MissionEventRecord]:
        with self._database.session_scope() as session:
            statement = (
                select(MissionEventRecord)
                .where(MissionEventRecord.mission_id == mission_id)
                .order_by(desc(MissionEventRecord.created_at))
            )
            if limit is not None:
                statement = statement.limit(limit)
            return list(session.scalars(statement))

    def record_approval(
        self,
        *,
        mission_id: str,
        step_id: str | None,
        decision: str,
        actor: str | None,
        reason: str | None,
        metadata: dict[str, Any],
    ) -> MissionApprovalRecordORM:
        with self._database.session_scope() as session:
            record = MissionApprovalRecordORM(
                mission_id=mission_id,
                step_id=step_id,
                decision=decision,
                actor=actor,
                reason=reason,
                metadata_json=_dumps(metadata),
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def list_approvals(self, mission_id: str) -> list[MissionApprovalRecordORM]:
        with self._database.session_scope() as session:
            statement = (
                select(MissionApprovalRecordORM)
                .where(MissionApprovalRecordORM.mission_id == mission_id)
                .order_by(desc(MissionApprovalRecordORM.created_at))
            )
            return list(session.scalars(statement))

    @staticmethod
    def load_snapshot(record: MissionRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return _loads(record.snapshot_json)

    @staticmethod
    def load_event_payload(record: MissionEventRecord) -> dict[str, Any]:
        return _loads(record.payload_json)

    @staticmethod
    def load_approval_metadata(record: MissionApprovalRecordORM) -> dict[str, Any]:
        return _loads(record.metadata_json)
