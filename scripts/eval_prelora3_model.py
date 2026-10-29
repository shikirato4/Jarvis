"""Run the existing PRE-LORA model cases with production registry retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy
from jarvis.bootstrap import build_application
from jarvis.config import Settings
from training.agentic_lora.evaluation.prelora2_dev_cases import CASES
from training.agentic_lora.evaluation.run_baseline import _managed_runtime
from training.agentic_lora.evaluation.run_prompt_abc import _run_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--arm", choices=("compact", "full"))
    args = parser.parse_args()
    manager, owned = _managed_runtime(Path.cwd())
    try:
        application = build_application(Settings(_env_file=None))
        schemas = application.action_executor.tool_schemas()
        semantics = application.action_executor.tool_semantics()
        rows = []
        output = Path("docs/evaluations/prelora3_candidate_model.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not args.case:
            output.write_text("", encoding="utf-8")
        for arm in ((args.arm,) if args.arm else ("compact", "full")):
            policy = AgenticPromptPolicy(profile=arm)
            for case in CASES:
                if args.case and case["id"] != args.case:
                    continue
                available = {item["function"]["name"]: item for item in schemas}
                for item in case["tools"]:
                    available.setdefault(item["function"]["name"], item)
                selected = policy.select_relevant_tools(
                    list(available.values()), case["user"], semantic_metadata=semantics
                )
                evaluation_case = {**case, "tools": selected}
                try:
                    row = _run_case(
                        "http://127.0.0.1:8081", "Qwen3.5-9B-IQ4_XS", 120,
                        arm, evaluation_case,
                    )
                except Exception as exc:
                    row = {"arm": arm, "case_id": case["id"], "runtime_error": type(exc).__name__}
                row["selected_schemas"] = [
                    item["function"]["name"] for item in selected
                ]
                row["retrieval_recall"] = set(case["sequence"]) <= set(row["selected_schemas"])
                rows.append(row)
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(arm, case["id"], row.get("tool_selection"), row.get("runtime_error"), flush=True)
                if "runtime_error" in row:
                    return
        for arm in ((args.arm,) if args.arm else ("compact", "full")):
            group = [row for row in rows if row["arm"] == arm]
            print(arm, {
                "exact_trajectory": sum(row["tool_selection"] for row in group) / len(group),
                "first_tool": sum(row["first_tool_correct"] for row in group) / len(group),
                "retrieval_recall": sum(row["retrieval_recall"] for row in group) / len(group),
                "arguments": sum(row["argument_valid"] for row in group) / len(group),
                "unnecessary_tools": sum(row["unnecessary_tool"] for row in group) / len(group),
                "false_execution": sum(row["false_execution"] for row in group) / len(group),
                "mean_prompt_tokens": statistics.mean(row["prompt_tokens_total"] for row in group),
                "mean_latency_ms": statistics.mean(row["total_latency_ms"] for row in group),
            })
    finally:
        if owned:
            manager.stop()
        manager.close()


if __name__ == "__main__":
    main()
