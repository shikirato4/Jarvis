from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import numpy as np


class NumpyMLPBackend:
    def __init__(
        self,
        *,
        labels: Sequence[str],
        input_dimension: int,
        hidden_dimension: int = 32,
        epochs: int = 160,
        learning_rate: float = 0.06,
        seed: int = 42,
    ) -> None:
        self._labels = tuple(labels)
        self._input_dimension = int(input_dimension)
        self._hidden_dimension = int(hidden_dimension)
        self._epochs = int(epochs)
        self._learning_rate = float(learning_rate)
        self._seed = int(seed)
        self._weights_1: np.ndarray | None = None
        self._bias_1: np.ndarray | None = None
        self._weights_2: np.ndarray | None = None
        self._bias_2: np.ndarray | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        if not features or len(features) != len(labels):
            raise ValueError("training features and labels must be non-empty and aligned")
        x = np.asarray(features, dtype=np.float64)
        label_index = {label: index for index, label in enumerate(self._labels)}
        y_index = np.asarray([label_index[label] for label in labels], dtype=np.int64)
        y = np.eye(len(self._labels), dtype=np.float64)[y_index]
        rng = np.random.default_rng(self._seed)
        self._weights_1 = rng.normal(0.0, 0.12, (self._input_dimension, self._hidden_dimension))
        self._bias_1 = np.zeros(self._hidden_dimension, dtype=np.float64)
        self._weights_2 = rng.normal(0.0, 0.12, (self._hidden_dimension, len(self._labels)))
        self._bias_2 = np.zeros(len(self._labels), dtype=np.float64)
        for _ in range(self._epochs):
            hidden_raw = x @ self._weights_1 + self._bias_1
            hidden = np.tanh(hidden_raw)
            logits = hidden @ self._weights_2 + self._bias_2
            probabilities = self._softmax(logits)
            gradient_logits = (probabilities - y) / len(x)
            gradient_w2 = hidden.T @ gradient_logits
            gradient_b2 = gradient_logits.sum(axis=0)
            gradient_hidden = (gradient_logits @ self._weights_2.T) * (1.0 - hidden * hidden)
            gradient_w1 = x.T @ gradient_hidden
            gradient_b1 = gradient_hidden.sum(axis=0)
            self._weights_2 -= self._learning_rate * gradient_w2
            self._bias_2 -= self._learning_rate * gradient_b2
            self._weights_1 -= self._learning_rate * gradient_w1
            self._bias_1 -= self._learning_rate * gradient_b1

    def predict_scores(self, features: list[float]) -> dict[str, float]:
        self._require_fitted()
        x = np.asarray(features, dtype=np.float64)
        hidden = np.tanh(x @ self._weights_1 + self._bias_1)
        probabilities = self._softmax(hidden @ self._weights_2 + self._bias_2)
        return {label: float(probabilities[index]) for index, label in enumerate(self._labels)}

    def save(self, path: Path) -> None:
        self._require_fitted()
        np.savez_compressed(
            path,
            labels=np.asarray(self._labels),
            weights_1=self._weights_1,
            bias_1=self._bias_1,
            weights_2=self._weights_2,
            bias_2=self._bias_2,
        )

    def load(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            labels = tuple(str(item) for item in checkpoint["labels"].tolist())
            if labels != self._labels:
                raise ValueError("router checkpoint labels do not match")
            self._weights_1 = checkpoint["weights_1"]
            self._bias_1 = checkpoint["bias_1"]
            self._weights_2 = checkpoint["weights_2"]
            self._bias_2 = checkpoint["bias_2"]

    def memory_bytes(self) -> int:
        arrays = (self._weights_1, self._bias_1, self._weights_2, self._bias_2)
        return sum(array.nbytes for array in arrays if array is not None)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=-1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=-1, keepdims=True)

    def _require_fitted(self) -> None:
        if self._weights_1 is None or self._weights_2 is None:
            raise RuntimeError("router backend is not fitted")

