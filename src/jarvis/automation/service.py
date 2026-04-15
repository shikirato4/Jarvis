from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, Field

from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.memory.service import AutomationEntry, MemoryService


class AutomationDefinition(BaseModel):
    automation_id: str | None = None
    name: str
    action_name: str
    payload: dict[str, Any] = Field(default_factory=dict)
    interval_seconds: int = 300
    enabled: bool = True


class AutomationService:
    def __init__(
        self,
        memory: MemoryService,
        router: ActionRouter,
        action_registry: ActionRegistry,
        timezone_name: str = "UTC",
        logger: logging.Logger | None = None,
    ) -> None:
        self._memory = memory
        self._router = router
        self._action_registry = action_registry
        self._logger = logger or logging.getLogger("jarvis.automation")
        self._scheduler = BackgroundScheduler(timezone=timezone_name)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        for automation in self._memory.list_automations(enabled_only=True):
            self._schedule_entry(automation)

    def stop(self) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False

    def save(self, definition: AutomationDefinition | dict[str, Any]) -> AutomationEntry:
        definition = AutomationDefinition.model_validate(definition)
        if self._action_registry.get(definition.action_name) is None:
            raise ValueError(f"action '{definition.action_name}' is not registered")
        entry = self._memory.save_automation(
            automation_id=definition.automation_id,
            name=definition.name,
            action_name=definition.action_name,
            payload=definition.payload,
            interval_seconds=definition.interval_seconds,
            enabled=definition.enabled,
        )
        if self._started:
            self._unschedule(entry.id)
            if entry.enabled:
                self._schedule_entry(entry)
        return entry

    def list(self, *, enabled_only: bool = False) -> list[AutomationEntry]:
        return self._memory.list_automations(enabled_only=enabled_only)

    def _schedule_entry(self, entry: AutomationEntry) -> None:
        self._scheduler.add_job(
            self._run_automation,
            trigger=IntervalTrigger(seconds=entry.interval_seconds),
            args=[entry.id],
            id=entry.id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        job = self._scheduler.get_job(entry.id)
        self._memory.update_automation_runtime(
            entry.id,
            last_run_at=entry.last_run_at,
            next_run_at=job.next_run_time if job else None,
        )

    def _unschedule(self, automation_id: str) -> None:
        if self._scheduler.get_job(automation_id):
            self._scheduler.remove_job(automation_id)

    def _run_automation(self, automation_id: str) -> None:
        entry = self._memory.get_automation(automation_id)
        if entry is None or not entry.enabled:
            self._unschedule(automation_id)
            return
        last_run_at = datetime.now(timezone.utc)
        try:
            self._router.execute(
                entry.action_name,
                entry.payload,
                metadata={"trigger": "automation", "automation_id": automation_id},
            )
        except Exception:
            self._logger.exception("automation_execution_failed", extra={"automation_id": automation_id})
        finally:
            job = self._scheduler.get_job(automation_id)
            self._memory.update_automation_runtime(
                automation_id,
                last_run_at=last_run_at,
                next_run_at=job.next_run_time if job else None,
            )
