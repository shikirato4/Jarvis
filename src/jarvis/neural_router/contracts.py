from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence


class RouterFeatureExtractor(Protocol):
    @property
    def dimension(self) -> int: ...

    def extract(self, text: str) -> list[float]: ...


class NeuralRouterBackend(Protocol):
    @property
    def labels(self) -> Sequence[str]: ...

    def fit(self, features: list[list[float]], labels: list[str]) -> None: ...

    def predict_scores(self, features: list[float]) -> dict[str, float]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def memory_bytes(self) -> int: ...

