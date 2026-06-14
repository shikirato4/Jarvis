from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, RLock


@dataclass
class DesktopMissionControl:
    pause_requested: Event = field(default_factory=Event)
    abort_requested: Event = field(default_factory=Event)


class DesktopAgentMissionCoordinator:
    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jarvis-desktop-agent")
        self._lock = RLock()
        self._controls: dict[str, DesktopMissionControl] = {}
        self._futures: dict[str, Future] = {}

    def control(self, mission_id: str) -> DesktopMissionControl:
        with self._lock:
            return self._controls.setdefault(mission_id, DesktopMissionControl())

    def submit(self, mission_id: str, fn, /, *args, **kwargs) -> Future:
        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._futures[mission_id] = future
        return future

    def future(self, mission_id: str) -> Future | None:
        with self._lock:
            return self._futures.get(mission_id)

    def request_pause(self, mission_id: str) -> None:
        self.control(mission_id).pause_requested.set()

    def request_resume(self, mission_id: str) -> None:
        self.control(mission_id).pause_requested.clear()

    def request_abort(self, mission_id: str) -> None:
        control = self.control(mission_id)
        control.abort_requested.set()
        control.pause_requested.clear()
        future = self.future(mission_id)
        if future is not None and not future.running():
            future.cancel()

    def clear_abort(self, mission_id: str) -> None:
        self.control(mission_id).abort_requested.clear()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
