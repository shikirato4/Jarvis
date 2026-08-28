from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable


EventHandler = Callable[[dict[str, Any]], None]
AllEventHandler = Callable[[str, dict[str, Any]], None]


class EventBus:
    def __init__(self, *, history_limit: int = 200) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._logger = logging.getLogger("jarvis.events")
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._all_handlers: list[AllEventHandler] = []
        self._lock = threading.RLock()

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[event_name].append(handler)

    def subscribe_all(self, handler: AllEventHandler) -> None:
        with self._lock:
            self._all_handlers.append(handler)

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._recent_events.appendleft({"event_name": event_name, "payload": payload, "published_at": datetime.now(timezone.utc)})
            handlers = list(self._handlers.get(event_name, []))
            all_handlers = list(self._all_handlers)
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                self._logger.exception("event_handler_failed", extra={"event_name": event_name})
        for handler in all_handlers:
            try:
                handler(event_name, payload)
            except Exception:
                self._logger.exception("event_handler_failed", extra={"event_name": event_name})

    def recent_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent_events)

    def trim(self, keep: int, *, max_age_seconds: float | None = None) -> int:
        with self._lock:
            before = len(self._recent_events)
            if max_age_seconds is not None:
                cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
                while self._recent_events and self._recent_events[-1]["published_at"].timestamp() < cutoff:
                    self._recent_events.pop()
            while len(self._recent_events) > keep:
                self._recent_events.pop()
            return max(before - len(self._recent_events), 0)
