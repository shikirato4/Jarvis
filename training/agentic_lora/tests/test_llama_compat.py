from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open

from training.agentic_lora.llama_compat import TARGET_TO_GGUF, build_structural_fixture, representative_targets
from training.agentic_lora.preflight import PINNED_MODEL_ID, PINNED_REVISION


ROOT = Path(__file__).parents[1]


def test_all_planned_targets_have_explicit_gguf_mapping() -> None:
    assert len(TARGET_TO_GGUF) == 12
    assert len(set(TARGET_TO_GGUF.values())) == 12


def test_representative_targets_resolve_text_linear_modules() -> None:
    targets = representative_targets(ROOT / "base_metadata")

    assert len(targets) == 12
    assert {target["logical_suffix"] for target in targets} == set(TARGET_TO_GGUF)
    assert all(target["gguf_tensor"].startswith("blk.") for target in targets)
    assert not any("vision" in target["hf_module"] for target in targets)


def test_structural_fixture_is_zero_effect_and_pinned(tmp_path: Path) -> None:
    output = tmp_path / "UNTRAINED_PREFLIGHT_ONLY"
    report = build_structural_fixture(ROOT / "base_metadata", output)
    config = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))

    assert report["fixture"] == "UNTRAINED_PREFLIGHT_ONLY"
    assert report["tensor_count"] == 24
    assert config["base_model_name_or_path"] == PINNED_MODEL_ID
    assert config["revision"] == PINNED_REVISION
    with safe_open(output / "adapter_model.safetensors", framework="pt") as tensors:
        assert len(tensors.keys()) == 24
        assert all(not tensors.get_tensor(name).count_nonzero() for name in tensors.keys())


def test_fixture_shapes_follow_peft_lora_orientation(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    report = build_structural_fixture(ROOT / "base_metadata", output, rank=2)
    first = report["targets"][0]
    prefix = f"base_model.model.{first['hf_module']}"

    with safe_open(output / "adapter_model.safetensors", framework="pt") as tensors:
        assert tensors.get_tensor(f"{prefix}.lora_A.weight").shape == (2, first["in_features"])
        assert tensors.get_tensor(f"{prefix}.lora_B.weight").shape == (first["out_features"], 2)


def test_fixture_can_isolate_one_target(tmp_path: Path) -> None:
    output = tmp_path / "fixture"
    report = build_structural_fixture(
        ROOT / "base_metadata",
        output,
        target_suffixes=["linear_attn.out_proj"],
    )

    assert report["tensor_count"] == 2
    assert [target["logical_suffix"] for target in report["targets"]] == ["linear_attn.out_proj"]


def test_compatibility_snapshot_is_pinned_and_complete() -> None:
    snapshot = json.loads((ROOT / "data" / "b4a1_compatibility_snapshot.json").read_text(encoding="utf-8"))

    assert snapshot["schema"] == "jarvis.agentic_lora.b4a1.compatibility.v1"
    assert snapshot["production"]["build"] == "b10516-b95502ba9"
    assert snapshot["production"]["gguf_sha256_before"] == snapshot["production"]["gguf_sha256_after"]
    assert len(snapshot["target_mapping"]) == 12
    assert {item["logical"] for item in snapshot["target_mapping"]} == set(TARGET_TO_GGUF)
    assert [item["logical"] for item in snapshot["target_mapping"] if item["official_conversion"] == "FAIL"] == [
        "linear_attn.out_proj"
    ]
    assert snapshot["runtime_smoke"]["exit_code"] == 0
    assert snapshot["cloud_started"] is False
    assert snapshot["training_started"] is False
