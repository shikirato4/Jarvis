from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from .contracts import NeuralRouterBackend
from .models import BenchmarkResult
from .training import should_promote


class RouterCheckpointStore(Protocol):
    def stage(self, backend: NeuralRouterBackend) -> None: ...

    def promote(self) -> None: ...

    def rollback(self) -> None: ...

    def load_active(self, backend: NeuralRouterBackend) -> None: ...


class FileRouterCheckpointStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._candidate = directory / "candidate.npz"
        self._active = directory / "active.npz"
        self._previous = directory / "previous.npz"

    def stage(self, backend: NeuralRouterBackend) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        backend.save(self._candidate)

    def promote(self) -> None:
        if not self._candidate.is_file():
            raise FileNotFoundError("router candidate checkpoint is missing")
        if self._active.is_file():
            shutil.copy2(self._active, self._previous)
        self._candidate.replace(self._active)

    def rollback(self) -> None:
        if not self._previous.is_file():
            raise FileNotFoundError("router rollback checkpoint is missing")
        shutil.copy2(self._previous, self._active)

    def load_active(self, backend: NeuralRouterBackend) -> None:
        backend.load(self._active)


class RouterLifecycleManager:
    def __init__(self, store: RouterCheckpointStore) -> None:
        self._store = store

    def stage_and_promote(
        self,
        backend: NeuralRouterBackend,
        candidate: BenchmarkResult,
        baseline: BenchmarkResult,
        *,
        min_macro_f1_improvement: float = 0.03,
    ) -> bool:
        if not should_promote(
            candidate,
            baseline,
            min_macro_f1_improvement=min_macro_f1_improvement,
        ):
            return False
        self._store.stage(backend)
        self._store.promote()
        return True

    def rollback(self, backend: NeuralRouterBackend) -> None:
        self._store.rollback()
        self._store.load_active(backend)

