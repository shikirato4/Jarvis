from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PINNED_MODEL_ID = "Qwen/Qwen3.5-9B"
PINNED_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
EXPECTED_TRAIN_SHA256 = "c48d05db88b85b2acb15bb5f8b178b7ad38434ca2498c7478a50cedf12af34bd"
EXPECTED_VALIDATION_SHA256 = "f4994819be616767bc91608162e858684ed5fb515c611b102acc0e1af31039d4"
EXCLUDED_MARKERS = ("embed_tokens", "lm_head", "norm", "vision", "visual", "projector", "audio")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_compiled(path: Path, *, expected_count: int, expected_hash: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    digest = sha256(path)
    if len(rows) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} records, found {len(rows)}")
    if digest != expected_hash:
        raise ValueError(f"{path}: SHA256 mismatch")
    samples = []
    for row in rows[:5]:
        labels = row["labels"]
        samples.append(
            {
                "record_id": row["record_id"],
                "input_tokens": len(row["input_ids"]),
                "labeled_tokens": sum(token != -100 for token in labels),
                "ignored_tokens": sum(token == -100 for token in labels),
            }
        )
    return {"count": len(rows), "sha256": digest, "mask_samples": samples}


def resolve_targets(model: Any, target_suffixes: list[str], *, rank: int) -> dict[str, Any]:
    modules = dict(model.named_modules())
    resolved: dict[str, list[dict[str, Any]]] = {}
    adapter_params = 0
    for suffix in target_suffixes:
        matches = []
        for name, module in modules.items():
            if not name.endswith(suffix):
                continue
            if type(module).__name__ != "Linear":
                raise TypeError(f"{name} is {type(module).__name__}, expected Linear")
            matches.append({"name": name, "type": type(module).__name__})
            adapter_params += rank * (module.in_features + module.out_features)
        if not matches:
            raise ValueError(f"missing target suffix: {suffix}")
        resolved[suffix] = matches
    selected_names = [item["name"] for matches in resolved.values() for item in matches]
    leaked = [name for name in selected_names if any(marker in name.casefold() for marker in EXCLUDED_MARKERS)]
    if leaked:
        raise ValueError(f"excluded module leakage: {leaked}")
    return {
        "patterns": {suffix: len(matches) for suffix, matches in resolved.items()},
        "module_count": len(selected_names),
        "adapter_parameter_estimate": adapter_params,
        "selected_names": selected_names,
        "excluded_leakage": leaked,
    }


def offline_preflight(root: Path) -> dict[str, Any]:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config_path = root / "configs" / "pilot_training.json"
    training_config = json.loads(config_path.read_text(encoding="utf-8"))
    if training_config["model_id"] != PINNED_MODEL_ID or training_config["revision"] != PINNED_REVISION:
        raise ValueError("pilot config is not pinned to the approved base revision")
    metadata = root / "base_metadata"
    wrapper_config = AutoConfig.from_pretrained(metadata, local_files_only=True)
    text_config = wrapper_config.text_config
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(text_config)
    targets = resolve_targets(model, training_config["target_modules"], rank=training_config["rank"])
    total_params = sum(parameter.numel() for parameter in model.parameters())
    compiled = root / "data" / "compiled"
    train = validate_compiled(compiled / "train.jsonl", expected_count=800, expected_hash=EXPECTED_TRAIN_SHA256)
    validation = validate_compiled(compiled / "validation.jsonl", expected_count=120, expected_hash=EXPECTED_VALIDATION_SHA256)
    return {
        "schema": "jarvis.agentic_lora.b4a.offline.v1",
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "wrapper_architecture": wrapper_config.architectures[0],
        "text_backbone_class": f"{type(model).__module__}.{type(model).__name__}",
        "text_backbone_path": "model (Qwen3_5ForCausalLM text-only instantiation from wrapper.text_config)",
        "text_config": {
            "model_type": text_config.model_type,
            "vocab_size": text_config.vocab_size,
            "hidden_size": text_config.hidden_size,
            "intermediate_size": text_config.intermediate_size,
            "num_hidden_layers": text_config.num_hidden_layers,
            "layer_types": list(text_config.layer_types),
            "dtype": str(text_config.dtype),
        },
        "total_parameter_estimate": total_params,
        "targets": targets,
        "lora": {"rank": 16, "alpha": 16, "dropout": 0.05, "bias": "none"},
        "train": train,
        "validation": validation,
        "optimizer_steps": 0,
        "cloud_execution_completed": False,
    }


def main() -> int:
    root = Path(__file__).parent
    report = offline_preflight(root)
    output = root / "data" / "b4a_offline_preflight.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
