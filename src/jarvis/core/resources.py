from __future__ import annotations

import os
import shutil
import threading
from collections import deque
from pathlib import Path
from typing import Any

from .models import ResourceSample

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None


class ResourceMonitor:
    def __init__(self, *, workspace_root: Path, data_dir: Path, history_limit: int = 120, poll_interval_seconds: float = 5.0, event_bus=None) -> None:
        self._workspace_root = workspace_root
        self._data_dir = data_dir
        self._history: deque[ResourceSample] = deque(maxlen=history_limit)
        self._poll_interval_seconds = max(poll_interval_seconds, 1.0)
        self._event_bus = event_bus
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._sample()
        self._thread = threading.Thread(target=self._run, name="jarvis-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> ResourceSample:
        if not self._history:
            self._sample()
        return self._history[0]

    def history(self) -> list[ResourceSample]:
        return list(self._history)

    def snapshot(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "latest": latest.model_dump(mode="json"),
            "history_count": len(self._history),
            "warnings": latest.warnings,
        }

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval_seconds):
            self._sample()

    def _sample(self) -> None:
        sample = ResourceSample()
        warnings: list[str] = []
        if psutil is not None:
            vm = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self._data_dir))
            process = psutil.Process(os.getpid())
            sample.cpu_percent = psutil.cpu_percent(interval=None)
            sample.process_cpu_percent = process.cpu_percent(interval=None)
            sample.ram_percent = float(vm.percent)
            sample.ram_used_bytes = int(vm.used)
            sample.ram_available_bytes = int(vm.available)
            sample.process_rss_bytes = int(process.memory_info().rss)
            sample.disk_total_bytes = int(disk.total)
            sample.disk_used_bytes = int(disk.used)
            sample.disk_free_bytes = int(disk.free)
            sample.disk_percent = float(disk.percent)
            if hasattr(psutil, "sensors_temperatures"):
                try:
                    temperatures = psutil.sensors_temperatures()
                    for readings in temperatures.values():
                        if readings:
                            sample.temperature_celsius = float(readings[0].current)
                            break
                except Exception:  # noqa: BLE001
                    pass
        else:
            disk = shutil.disk_usage(self._data_dir)
            sample.disk_total_bytes = int(disk.total)
            sample.disk_used_bytes = int(disk.used)
            sample.disk_free_bytes = int(disk.free)
            sample.disk_percent = round((disk.used / disk.total) * 100, 2) if disk.total else None
            warnings.extend(["psutil_unavailable_cpu", "psutil_unavailable_ram"])
        if sample.ram_percent is not None and sample.ram_percent >= 90:
            warnings.append("high_ram_pressure")
        if sample.disk_percent is not None and sample.disk_percent >= 90:
            warnings.append("high_disk_pressure")
        if sample.cpu_percent is not None and sample.cpu_percent >= 90:
            warnings.append("high_cpu_pressure")
        sample.warnings = warnings
        sample.metadata = {
            "workspace_root": str(self._workspace_root),
            "data_dir": str(self._data_dir),
        }
        self._history.appendleft(sample)
        if self._event_bus is not None and warnings:
            self._event_bus.publish("ops.resources.warning", sample.model_dump(mode="json"))
