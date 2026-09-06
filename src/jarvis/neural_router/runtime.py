from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from .models import RouterExample, RouterPrediction, ShadowTrace, TrainingResult
from .router import ShadowRouter
from .training import NeuralRouterTrainingPipeline


class LazyTrainingShadowRouter:
    """Train once on first use and observe without influencing selected tools.

    The runtime accepts the backend-neutral training pipeline. NumPy, PyTorch, or
    another backend can therefore be supplied by composition without changing
    the agent controller or this lifecycle wrapper.
    """

    def __init__(
        self,
        *,
        pipeline: NeuralRouterTrainingPipeline,
        dataset_factory: Callable[[], list[RouterExample]],
        seed: int = 42,
        trace_limit: int = 1_000,
    ) -> None:
        self._pipeline = pipeline
        self._dataset_factory = dataset_factory
        self._seed = seed
        self._trace_limit = trace_limit
        self._lock = Lock()
        self._result: TrainingResult | None = None
        self._shadow: ShadowRouter | None = None

    @property
    def training_result(self) -> TrainingResult | None:
        return self._result

    @property
    def traces(self) -> list[ShadowTrace]:
        shadow = self._shadow
        return [] if shadow is None else list(shadow.traces)

    @property
    def promoted(self) -> bool:
        return bool(self._result is not None and self._result.promoted)

    def predict(self, text: str, *, top_k: int = 3) -> RouterPrediction:
        return self._ensure_shadow().predict(text, top_k=top_k)

    def tools_for_intent(self, intent: str) -> list[str]:
        return self._ensure_shadow().tools_for_intent(intent)

    def observe(
        self,
        *,
        text: str,
        selected_tools: list[str],
        success: bool,
        step_count: int,
        latency_ms: float,
        correction_required: bool,
    ) -> list[str]:
        shadow = self._ensure_shadow()
        return shadow.observe(
            text=text,
            selected_tools=selected_tools,
            success=success,
            step_count=step_count,
            latency_ms=latency_ms,
            correction_required=correction_required,
        )

    def _ensure_shadow(self) -> ShadowRouter:
        if self._shadow is not None:
            return self._shadow
        with self._lock:
            if self._shadow is None:
                self._result = self._pipeline.train_and_evaluate(
                    self._dataset_factory(),
                    seed=self._seed,
                )
                self._shadow = ShadowRouter(
                    self._result.router,
                    trace_limit=self._trace_limit,
                )
        return self._shadow
