from __future__ import annotations

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class RouterExample(JarvisBaseModel):
    text: str
    intent: str
    tools: tuple[str, ...] = ()
    source: str = "controlled_synthetic"


class DatasetSplit(JarvisBaseModel):
    train: list[RouterExample]
    validation: list[RouterExample]
    test: list[RouterExample]


class RouterScore(JarvisBaseModel):
    label: str
    score: float


class RouterPrediction(JarvisBaseModel):
    intent: str
    scores: list[RouterScore]
    candidate_tools: list[str] = Field(default_factory=list)
    needs_agent_mode: bool = False


class BenchmarkResult(JarvisBaseModel):
    accuracy: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    top3_recall: float = 0.0
    latency_ms: float = 0.0


class TrainingResult(JarvisBaseModel):
    validation: BenchmarkResult
    test: BenchmarkResult
    majority_baseline: BenchmarkResult
    rule_baseline: BenchmarkResult
    memory_bytes: int = 0
    promoted: bool = False
    backend_name: str = ""
    router: object = Field(exclude=True, repr=False)


class ShadowTrace(JarvisBaseModel):
    intent_class: str
    candidate_tools: list[str]
    selected_tools: list[str]
    success: bool
    step_count: int
    latency_ms: float
    correction_required: bool

