from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from training.agentic_lora.preflight import PINNED_MODEL_ID, PINNED_REVISION


TARGET_TO_GGUF: dict[str, str] = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.in_proj_b": "ssm_beta",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}

EXCLUDED_MARKERS = ("embed_tokens", "lm_head", "norm", "vision", "audio", "projector")


def representative_targets(metadata: Path) -> list[dict[str, Any]]:
    wrapper_config = AutoConfig.from_pretrained(metadata, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(wrapper_config.text_config)
    modules = dict(model.named_modules())
    selected: list[dict[str, Any]] = []
    for suffix, gguf_stem in TARGET_TO_GGUF.items():
        matches = [(name, module) for name, module in modules.items() if name.endswith(suffix)]
        if not matches:
            raise ValueError(f"missing target suffix: {suffix}")
        name, module = matches[0]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} is {type(module).__name__}, expected Linear")
        if any(marker in name.casefold() for marker in EXCLUDED_MARKERS):
            raise ValueError(f"excluded target selected: {name}")
        layer = int(name.split(".layers.", 1)[1].split(".", 1)[0])
        selected.append(
            {
                "logical_suffix": suffix,
                "hf_module": name,
                "gguf_tensor": f"blk.{layer}.{gguf_stem}.weight",
                "in_features": module.in_features,
                "out_features": module.out_features,
            }
        )
    return selected


def build_structural_fixture(
    metadata: Path,
    output_dir: Path,
    *,
    rank: int = 1,
    target_suffixes: Iterable[str] | None = None,
) -> dict[str, Any]:
    if rank <= 0:
        raise ValueError("rank must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    requested = list(target_suffixes) if target_suffixes is not None else list(TARGET_TO_GGUF)
    unknown = sorted(set(requested) - TARGET_TO_GGUF.keys())
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    selected = [
        target for target in representative_targets(metadata) if target["logical_suffix"] in requested
    ]
    if len(selected) != len(requested):
        raise ValueError("target suffixes must be unique")
    tensors: dict[str, torch.Tensor] = {}
    for target in selected:
        prefix = f"base_model.model.{target['hf_module']}"
        tensors[f"{prefix}.lora_A.weight"] = torch.zeros(
            (rank, target["in_features"]), dtype=torch.bfloat16
        )
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
            (target["out_features"], rank), dtype=torch.bfloat16
        )
    save_file(tensors, output_dir / "adapter_model.safetensors")
    config = {
        "base_model_name_or_path": PINNED_MODEL_ID,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "revision": PINNED_REVISION,
        "target_modules": requested,
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "fixture": "UNTRAINED_PREFLIGHT_ONLY",
        "rank": rank,
        "targets": selected,
        "tensor_count": len(tensors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a zero-effect structural PEFT LoRA fixture")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--target", action="append", choices=list(TARGET_TO_GGUF))
    args = parser.parse_args()
    report = build_structural_fixture(
        args.metadata,
        args.output_dir,
        rank=args.rank,
        target_suffixes=args.target,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
