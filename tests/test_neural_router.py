from __future__ import annotations

from jarvis.neural_router import (
    BenchmarkResult,
    HashedTextFeatureExtractor,
    NeuralRouter,
    NeuralRouterTrainingPipeline,
    NumpyMLPBackend,
    RouterLifecycleManager,
    RouterExample,
    ShadowRouter,
    build_controlled_dataset,
    should_promote,
    stratified_split,
)


class _BackendStub:
    labels = ("chat", "calculator", "web")

    def __init__(self) -> None:
        self.features = None

    def fit(self, features, labels) -> None:
        self.features = features

    def predict_scores(self, features):
        assert isinstance(features, list)
        return {"chat": 0.1, "calculator": 0.8, "web": 0.1}

    def save(self, path) -> None:
        return

    def load(self, path) -> None:
        return

    def memory_bytes(self) -> int:
        return 0


def test_router_contract_does_not_expose_numpy_to_callers() -> None:
    backend = _BackendStub()
    router = NeuralRouter(
        backend=backend,
        feature_extractor=HashedTextFeatureExtractor(dimension=64),
        tool_map={"calculator": ("calculator.evaluate",)},
    )

    prediction = router.predict("cuánto es 27 por 83", top_k=3)

    assert prediction.intent == "calculator"
    assert prediction.candidate_tools == ["calculator.evaluate"]
    assert prediction.scores[0].label == "calculator"
    assert isinstance(router.feature_extractor.extract("hola"), list)


def test_controlled_dataset_is_stratified_and_contains_no_private_traces() -> None:
    dataset = build_controlled_dataset()
    split = stratified_split(dataset, seed=721)

    labels = {example.intent for example in dataset}
    assert len(labels) >= 10
    assert labels <= {example.intent for example in split.train}
    assert labels <= {example.intent for example in split.validation}
    assert labels <= {example.intent for example in split.test}
    assert all("api_key" not in example.text.casefold() for example in dataset)
    assert all(example.source == "controlled_synthetic" for example in dataset)


def test_numpy_backend_training_beats_majority_and_can_be_promoted() -> None:
    pipeline = NeuralRouterTrainingPipeline(
        backend_factory=lambda labels, dimension: NumpyMLPBackend(
            labels=labels,
            input_dimension=dimension,
            hidden_dimension=32,
            epochs=180,
            learning_rate=0.08,
            seed=42,
        ),
        feature_extractor=HashedTextFeatureExtractor(dimension=256),
    )

    result = pipeline.train_and_evaluate(build_controlled_dataset(), seed=42)

    assert result.test.accuracy >= 0.70
    assert result.test.macro_f1 >= 0.65
    assert result.test.top3_recall >= result.test.accuracy
    assert result.test.accuracy > result.majority_baseline.accuracy + 0.40
    assert result.memory_bytes > 0
    assert should_promote(
        result.test,
        result.rule_baseline,
        min_macro_f1_improvement=0.0,
    ) is (result.test.macro_f1 > result.rule_baseline.macro_f1)


def test_shadow_mode_never_controls_selected_tool_or_stores_prompt() -> None:
    router = NeuralRouter(
        backend=_BackendStub(),
        feature_extractor=HashedTextFeatureExtractor(dimension=32),
        tool_map={"calculator": ("calculator.evaluate",)},
    )
    shadow = ShadowRouter(router)
    secret_prompt = "calcula 20 por 10 SECRET-TRACE-991"

    selected = shadow.observe(
        text=secret_prompt,
        selected_tools=["web.search"],
        success=False,
        step_count=2,
        latency_ms=30.0,
        correction_required=True,
    )

    assert selected == ["web.search"]
    assert shadow.traces[0].candidate_tools == ["calculator.evaluate"]
    assert shadow.traces[0].selected_tools == ["web.search"]
    assert secret_prompt not in repr(shadow.traces)


def test_promotion_rejects_candidate_without_clear_improvement() -> None:
    candidate = BenchmarkResult(accuracy=0.8, macro_f1=0.79)
    baseline = BenchmarkResult(accuracy=0.81, macro_f1=0.80)

    assert should_promote(candidate, baseline) is False


def test_checkpoint_promotion_and_rollback_use_backend_neutral_store() -> None:
    calls = []

    class Store:
        def stage(self, backend):
            calls.append(("stage", type(backend).__name__))

        def promote(self):
            calls.append(("promote",))

        def rollback(self):
            calls.append(("rollback",))

        def load_active(self, backend):
            calls.append(("load_active", type(backend).__name__))

    backend = _BackendStub()
    lifecycle = RouterLifecycleManager(Store())
    better = BenchmarkResult(accuracy=0.9, macro_f1=0.9)
    baseline = BenchmarkResult(accuracy=0.7, macro_f1=0.7)

    assert lifecycle.stage_and_promote(backend, better, baseline) is True
    lifecycle.rollback(backend)

    assert calls == [
        ("stage", "_BackendStub"),
        ("promote",),
        ("rollback",),
        ("load_active", "_BackendStub"),
    ]
