from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from time import perf_counter

from .contracts import NeuralRouterBackend, RouterFeatureExtractor
from .dataset import stratified_split
from .models import BenchmarkResult, RouterExample, TrainingResult
from .router import NeuralRouter


BackendFactory = Callable[[tuple[str, ...], int], NeuralRouterBackend]


class NeuralRouterTrainingPipeline:
    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        feature_extractor: RouterFeatureExtractor,
    ) -> None:
        self._backend_factory = backend_factory
        self._features = feature_extractor

    def train_and_evaluate(
        self,
        examples: list[RouterExample],
        *,
        seed: int = 42,
    ) -> TrainingResult:
        split = stratified_split(examples, seed=seed)
        labels = tuple(sorted({example.intent for example in examples}))
        tool_map: dict[str, tuple[str, ...]] = {}
        for example in examples:
            tool_map.setdefault(example.intent, example.tools)
        backend = self._backend_factory(labels, self._features.dimension)
        backend.fit(
            [self._features.extract(example.text) for example in split.train],
            [example.intent for example in split.train],
        )
        router = NeuralRouter(
            backend=backend,
            feature_extractor=self._features,
            tool_map=tool_map,
        )
        validation = evaluate_router(router, split.validation)
        test = evaluate_router(router, split.test)
        majority_label = Counter(example.intent for example in split.train).most_common(1)[0][0]
        majority = evaluate_predictions(
            [example.intent for example in split.test],
            [[majority_label] for _ in split.test],
        )
        rule_predictions = [rule_rank(example.text, labels) for example in split.test]
        rule = evaluate_predictions(
            [example.intent for example in split.test],
            rule_predictions,
        )
        promoted = should_promote(test, rule)
        return TrainingResult(
            validation=validation,
            test=test,
            majority_baseline=majority,
            rule_baseline=rule,
            memory_bytes=backend.memory_bytes(),
            promoted=promoted,
            backend_name=type(backend).__name__,
            router=router,
        )


def evaluate_router(router: NeuralRouter, examples: list[RouterExample]) -> BenchmarkResult:
    started = perf_counter()
    predictions = [
        [score.label for score in router.predict(example.text, top_k=3).scores]
        for example in examples
    ]
    elapsed_ms = (perf_counter() - started) * 1_000
    result = evaluate_predictions([example.intent for example in examples], predictions)
    return result.model_copy(update={"latency_ms": elapsed_ms / max(len(examples), 1)})


def evaluate_predictions(actual: list[str], ranked: list[list[str]]) -> BenchmarkResult:
    labels = sorted(set(actual) | {item for prediction in ranked for item in prediction})
    top1 = [prediction[0] if prediction else "" for prediction in ranked]
    accuracy = sum(left == right for left, right in zip(actual, top1)) / max(len(actual), 1)
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(a == label and p == label for a, p in zip(actual, top1))
        false_positive = sum(a != label and p == label for a, p in zip(actual, top1))
        false_negative = sum(a == label and p != label for a, p in zip(actual, top1))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
    top3 = sum(target in prediction[:3] for target, prediction in zip(actual, ranked)) / max(len(actual), 1)
    return BenchmarkResult(
        accuracy=accuracy,
        macro_f1=sum(f1_values) / max(len(f1_values), 1),
        macro_precision=sum(precisions) / max(len(precisions), 1),
        macro_recall=sum(recalls) / max(len(recalls), 1),
        top3_recall=top3,
    )


def rule_rank(text: str, labels: tuple[str, ...]) -> list[str]:
    lowered = text.casefold()
    keywords = {
        "calculator": ("cuánto", "calcula", "raíz", "porcentaje"),
        "web": ("noticias", "hoy", "web", "fuentes", "actual"),
        "data": ("csv", "xlsx", "tabla", "datos", "gráfica", "anomal"),
        "vision": ("foto", "imagen", "pantalla", "visible"),
        "image": ("genera una imagen", "ilustración", "dibuja", "arte"),
        "code": ("código", "python", "bug", "tests", "repo"),
        "files": ("archivo", "carpeta", "pdf", "disco"),
        "windows": ("ventana", "spotify", "notepad", "click", "aplicación"),
        "git": ("git", "diff", "ramas", "commit"),
        "memory": ("recuerda", "memoria", "olvida"),
        "science": ("derivada", "integral", "órbita", "matriz"),
        "audio": ("audio", "mp3", "wav", "voz"),
        "video": ("video", "mp4", "fotogramas", "clip"),
        "agent": ("varios pasos", "misión", "varias herramientas", "por tu cuenta"),
    }
    scored = sorted(
        labels,
        key=lambda label: sum(marker in lowered for marker in keywords.get(label, ())),
        reverse=True,
    )
    return scored[:3]


def should_promote(
    candidate: BenchmarkResult,
    baseline: BenchmarkResult,
    *,
    min_macro_f1_improvement: float = 0.03,
) -> bool:
    return (
        candidate.macro_f1 >= baseline.macro_f1 + min_macro_f1_improvement
        and candidate.accuracy >= baseline.accuracy
    )

