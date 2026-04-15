from __future__ import annotations

import logging
from collections import defaultdict
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable


EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    def __init__(self, *, history_limit: int = 200) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._logger = logging.getLogger("jarvis.events")
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=history_limit)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        self._recent_events.appendleft({"event_name": event_name, "payload": payload, "published_at": datetime.now(timezone.utc)})
        for handler in self._handlers.get(event_name, []):
            try:
                handler(payload)
            except Exception:
                self._logger.exception("event_handler_failed", extra={"event_name": event_name})

    def recent_events(self) -> list[dict[str, Any]]:
        return list(self._recent_events)

    def trim(self, keep: int, *, max_age_seconds: float | None = None) -> int:
        before = len(self._recent_events)
        if max_age_seconds is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
            while self._recent_events and self._recent_events[-1]["published_at"].timestamp() < cutoff:
                self._recent_events.pop()
        while len(self._recent_events) > keep:
            self._recent_events.pop()
        return max(before - len(self._recent_events), 0)
