from __future__ import annotations

from typing import Any

from .views import HudTimelineEntry


class HudTimelineComposer:
    def __init__(self, event_bus, state_manager) -> None:
        self._event_bus = event_bus
        self._state_manager = state_manager

    def build(self, *, limit: int = 50) -> list[HudTimelineEntry]:
        snapshot = self._state_manager.snapshot(action_names=[], tool_names=[], include_history=True)
        entries: list[HudTimelineEntry] = []
        for event in self._event_bus.recent_events()[:limit]:
            entries.append(
                HudTimelineEntry(
                    entry_type="event",
                    service_name=self._infer_service_name(event["event_name"]),
                    title=event["event_name"],
                    status=str(event["payload"].get("status", "published")),
                    timestamp=event["published_at"],
                    data=event["payload"],
                )
            )
        for record in snapshot.recent_tasks[:limit]:
            entries.append(
                HudTimelineEntry(
                    entry_type="task",
                    service_name=record.source,
                    title=f"{record.route_type}:{record.target}",
                    status=record.status.value,
                    timestamp=record.received_at,
                    data=record.model_dump(mode="json"),
                )
            )
        for record in snapshot.recent_autonomy_receipts[:limit]:
            entries.append(
                HudTimelineEntry(
                    entry_type="autonomy",
                    service_name="autonomy",
                    title=record.operation_name,
                    status=record.status,
                    timestamp=record.invoked_at,
                    data=record.model_dump(mode="json"),
                )
            )
        for record in snapshot.recent_voice_invocations[:limit]:
            entries.append(
                HudTimelineEntry(
                    entry_type="voice",
                    service_name="voice_runtime",
                    title=record.operation_name,
                    status=record.status,
                    timestamp=record.invoked_at,
                    data=record.model_dump(mode="json"),
                )
            )
        for record in snapshot.recent_vision_invocations[:limit]:
            entries.append(
                HudTimelineEntry(
                    entry_type="vision",
                    service_name="vision_runtime",
                    title=record.operation_name,
                    status=record.status,
                    timestamp=record.invoked_at,
                    data=record.model_dump(mode="json"),
                )
            )
        entries.sort(key=lambda item: item.timestamp, reverse=True)
        return entries[:limit]

    @staticmethod
    def _infer_service_name(event_name: str) -> str:
        if "." not in event_name:
            return event_name
        return event_name.split(".", 1)[0]
