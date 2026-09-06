from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from training.agentic_lora.preflight import (
    EXPECTED_TRAIN_SHA256,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    offline_preflight,
    resolve_targets,
    validate_compiled,
)


ROOT = Path(__file__).parents[1]


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.proj = nn.Linear(4, 6, bias=False)
        self.model.embed_tokens = nn.Linear(4, 4, bias=False)


def test_offline_preflight_pins_identity_targets_and_split_counts() -> None:
    report = offline_preflight(ROOT)

    assert report["model_id"] == PINNED_MODEL_ID
    assert report["revision"] == PINNED_REVISION
    assert report["train"]["count"] == 800
    assert report["validation"]["count"] == 120
    assert report["targets"]["module_count"] == 248
    assert report["targets"]["excluded_leakage"] == []
    assert report["optimizer_steps"] == 0


def test_compiled_hash_is_stable_and_masking_is_nonempty() -> None:
    report = validate_compiled(
        ROOT / "data" / "compiled" / "train.jsonl",
        expected_count=800,
        expected_hash=EXPECTED_TRAIN_SHA256,
    )

    assert report["sha256"] == EXPECTED_TRAIN_SHA256
    assert all(sample["labeled_tokens"] > 0 for sample in report["mask_samples"])
    assert all(sample["ignored_tokens"] > 0 for sample in report["mask_samples"])


def test_target_resolver_rejects_missing_and_excluded_modules() -> None:
    model = _TinyModel()

    with pytest.raises(ValueError, match="missing target"):
        resolve_targets(model, ["missing"], rank=2)
    with pytest.raises(ValueError, match="excluded module leakage"):
        resolve_targets(model, ["embed_tokens"], rank=2)


def test_target_parameter_estimate_is_reproducible() -> None:
    report = resolve_targets(_TinyModel(), ["proj"], rank=2)

    assert report["module_count"] == 1
    assert report["adapter_parameter_estimate"] == 20
