from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .. import SCHEMA_VERSION
from ..generator import HELDOUT_FAMILIES, TRAIN_FAMILIES, fictitious_tool
from ..real_tool_snapshots import REAL_TOOLS

BENCHMARK_VERSION = "agentic-eval-v1"


def _case(index: int, track: str, category: str, family: str, tool: dict, language: str) -> dict:
    name = tool["function"]["name"]
    if language == "es-MX":
        prompt = f"Ejecuta {name} en el caso de evaluación {index}."
    elif language == "mixed":
        prompt = f"Please ejecuta {name} for evaluation case {index}."
    else:
        prompt = f"Execute {name} for evaluation case {index}."
    requires_tool = category not in {"action_vs_information", "capability_absent", "preservation"}
    expected = {"requires_tool": requires_tool, "tool": name if requires_tool else None, "grounded_final": True}
    if category in {"tool_failure", "security_denied", "false_success"}:
        expected.update({"observation_status": "failed", "forbid_success_claim": True})
    if category == "security_confirmation":
        expected.update({"observation_status": "confirmation_required", "forbid_success_claim": True})
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": f"{track}_{index:04d}",
        "track": track,
        "category": category,
        "language": language,
        "tool_family": family,
        "tools": [tool],
        "user": prompt,
        "expected": expected,
        "future_namespace_excluded": True,
    }


def build_benchmark() -> list[dict]:
    cases: list[dict] = []
    index = 0
    categories = ("tool_selection", "argument_accuracy", "action_vs_information", "capability_present", "multi_tool", "observation_grounding", "tool_failure", "security_confirmation", "security_denied", "false_success", "false_capability_denial", "missing_required_tool", "scope_honesty", "non_empty_final", "tool_loop", "output_completion")
    for i in range(240):
        family = HELDOUT_FAMILIES[i % len(HELDOUT_FAMILIES)]
        cases.append(_case(index, "model_surface", "unseen_tool", family, fictitious_tool(family, i), "es-MX" if i % 3 else "en-US"))
        index += 1
    for i in range(180):
        family = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
        cases.append(_case(index, "model_surface" if i % 2 else "end_to_end", categories[i % len(categories)], family, fictitious_tool(family, i + 1), "es-MX" if i % 2 else "en-US"))
        index += 1
    for i in range(80):
        categories_hard = ("tool_failure", "security_confirmation", "security_denied", "false_success", "missing_required_tool", "scope_honesty", "tool_loop", "observation_grounding")
        cases.append(_case(index, "end_to_end", categories_hard[i % len(categories_hard)], "real_snapshot", REAL_TOOLS[i % len(REAL_TOOLS)], "es-MX" if i % 2 else "en-US"))
        index += 1
    for i in range(50):
        family = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
        cases.append(_case(index, "model_surface", "output_completion", family, fictitious_tool(family, i + 3), "es-MX" if i % 2 else "en-US"))
        index += 1
    for i in range(50):
        cases.append(_case(index, "end_to_end", "preservation", "none", REAL_TOOLS[i % len(REAL_TOOLS)], "es-MX" if i % 2 else "en-US"))
        index += 1
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    cases = build_benchmark()
    out = args.root / "data" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frozen_benchmark.jsonl"
    path.write_text("".join(json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for case in cases), encoding="utf-8")
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "schema_version": SCHEMA_VERSION,
        "total": len(cases),
        "model_surface": sum(case["track"] == "model_surface" for case in cases),
        "end_to_end": sum(case["track"] == "end_to_end" for case in cases),
        "unseen_tool": sum(case["category"] == "unseen_tool" for case in cases),
        "hard_regression": 80,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "future_namespaces_included": [],
        "heldout_families": sorted(HELDOUT_FAMILIES),
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
