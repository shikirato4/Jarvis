"""Isolate model cases so a local server reset cannot erase A/B evidence."""

from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

from training.agentic_lora.evaluation.prelora2_dev_cases import CASES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output = Path("docs/evaluations/prelora3_candidate_model.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        output.write_text("", encoding="utf-8")
        for arm in ("compact", "full"):
            for case in CASES:
                command = [
                    sys.executable, "-m", "scripts.eval_prelora3_model",
                    "--case", case["id"], "--arm", arm,
                ]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                matched = [
                    row for row in rows
                    if row.get("arm") == arm and row.get("case_id") == case["id"]
                ]
                if not matched:
                    error = {"arm": arm, "case_id": case["id"], "runtime_error": "ProcessExit"}
                    with output.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(error) + "\n")
                    matched = [error]
                print(arm, case["id"], matched[-1].get("tool_selection"),
                      matched[-1].get("runtime_error"), completed.returncode, flush=True)
    latest = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[(row["arm"], row["case_id"])] = row
    rows = list(latest.values())
    summary = {}
    for arm in ("compact", "full"):
        group = [row for row in rows if row.get("arm") == arm]
        summary[arm] = {
            "cases": len(group),
            "exact_trajectory": sum(bool(row.get("tool_selection")) for row in group) / len(group),
            "first_tool": sum(bool(row.get("first_tool_correct")) for row in group) / len(group),
            "retrieval_recall": sum(bool(row.get("retrieval_recall")) for row in group) / len(group),
            "runtime_failures": sum(bool(row.get("runtime_error")) for row in group),
            "prompt_tokens": sum(row.get("prompt_tokens_total", 0) for row in group) / len(group),
            "latency_ms": sum(row.get("total_latency_ms", 0) for row in group) / len(group),
        }
    summary_path = output.with_name("prelora3_candidate_model_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
