from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).parents[1]
    data = root / "data"
    evaluation = data / "evaluation"
    benchmark = evaluation / "frozen_benchmark.jsonl"
    results = evaluation / "baseline_results.jsonl"
    summary = evaluation / "baseline_summary.json"
    manifest = {
        "freeze_schema": "jarvis.agentic_lora.base_freeze.v1",
        "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": 600,
        "benchmark_sha256": sha(benchmark),
        "results_sha256": sha(results),
        "summary_sha256": sha(summary),
        "model": "Qwen3.5-9B-IQ4_XS",
        "profile": "qwen35-9b-iq4-xs",
        "gguf_sha256": "7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429",
        "runtime": {"context": 8192, "parallel": 1, "thinking": "OFF", "temperature": 0, "top_p": 1, "max_tokens": 768, "llama_cpp": "b10516-b95502ba9", "provider": "qwen_local"},
        "scoring_version": "agentic-eval-v1-score-v1",
        "critical_metrics": json.loads(summary.read_text(encoding="utf-8")),
        "benchmark_manifest_sha256": sha(evaluation / "benchmark_manifest.json"),
    }
    path = evaluation / "base_freeze_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
