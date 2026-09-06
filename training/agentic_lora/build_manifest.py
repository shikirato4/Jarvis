from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).parent
    data = root / "data"
    canonical = data / "canonical"
    compiled = data / "compiled"
    evaluation = data / "evaluation"
    manifest = {
        "manifest_version": "b3-manifest-v1",
        "schema_version": "jarvis.agentic_lora.v1",
        "generator_version": "agentic-pilot-v1",
        "compiler_version": "compiler-v1",
        "seed": 20260831,
        "base_model": {
            "id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "production_profile": "qwen35-9b-iq4-xs",
            "production_gguf": "models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf",
            "full_bf16_downloaded_locally": False,
        },
        "hashes": {
            "pilot_config": digest(root / "configs" / "pilot.json"),
            "training_config": digest(root / "configs" / "pilot_training.json"),
            "canonical_train": digest(canonical / "train.jsonl"),
            "canonical_validation": digest(canonical / "validation.jsonl"),
            "canonical_heldout": digest(canonical / "heldout.jsonl"),
            "canonical_hard_regression": digest(canonical / "hard_regression.jsonl"),
            "compiled_train": digest(compiled / "train.jsonl"),
            "compiled_validation": digest(compiled / "validation.jsonl"),
            "compiled_heldout": digest(compiled / "heldout.jsonl"),
            "compiled_hard_regression": digest(compiled / "hard_regression.jsonl"),
            "evaluation_benchmark": digest(evaluation / "frozen_benchmark.jsonl"),
            "tokenization_report": digest(compiled / "tokenization_report.json"),
            "tokenizer_identity": digest(root / "base_metadata" / "identity_manifest.json"),
            "parity_report": digest(data / "parity_report.json"),
            "spot_audit": digest(data / "spot_audit.json"),
        },
        "baseline": {
            "status": "partial_resumable",
            "results": "data/evaluation/baseline_results.jsonl",
            "summary": "data/evaluation/baseline_summary.json",
            "model": "Qwen3.5-9B-IQ4_XS",
            "context": 8192,
            "parallel": 1,
            "thinking": "off",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 768,
            "endpoint": "http://127.0.0.1:8081",
            "scoring_version": "agentic-eval-v1-score-v1",
            "results_sha256": digest(evaluation / "baseline_results.jsonl") if (evaluation / "baseline_results.jsonl").exists() else None,
            "summary_sha256": digest(evaluation / "baseline_summary.json") if (evaluation / "baseline_summary.json").exists() else None,
        },
        "training_environment": {
            "platform": "Linux CUDA cloud, not provisioned in B3",
            "target_gpu": "48 GB NVIDIA GPU",
            "requirements": "configs/requirements-linux-cu130.txt",
        },
    }
    path = root / "data" / "b3_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
