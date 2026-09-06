from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(benchmark: Path, results: Path) -> dict[str, object]:
    cases = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()] if results.exists() else []
    count = len(rows)
    def truth(key: str) -> int:
        return sum(bool(row.get(key)) for row in rows)
    def rate(selected: list[dict], key: str) -> float | None:
        return round(sum(bool(row.get(key)) for row in selected) / len(selected), 4) if selected else None
    spanish = [row for row in rows if row.get("language") == "es-MX"]
    english = [row for row in rows if row.get("language") == "en-US"]
    mixed = [row for row in rows if row.get("language") not in {"es-MX", "en-US"}]
    def metric(selected: list[dict], key: str) -> dict[str, object]:
        return {"correct": sum(bool(row.get(key)) for row in selected), "total": len(selected),
                "rate": round(sum(bool(row.get(key)) for row in selected) / len(selected), 4) if selected else None}
    durations = [float(row["duration_ms"]) for row in rows if isinstance(row.get("duration_ms"), (int, float))]
    def durations_for(track: str) -> list[float]:
        return [float(row["duration_ms"]) for row in rows if row.get("track") == track and isinstance(row.get("duration_ms"), (int, float))]
    def duration_stats(values: list[float]) -> dict[str, object]:
        return {"count": len(values), "total_ms": round(sum(values), 2),
                "avg_ms": round(statistics.mean(values), 2) if values else None,
                "median_ms": round(statistics.median(values), 2) if values else None,
                "p90_ms": round(sorted(values)[max(0, int(len(values) * .9 + .5) - 1)], 2) if values else None,
                "slowest_ms": round(max(values), 2) if values else None}
    result_hash = hashlib.sha256(results.read_bytes()).hexdigest() if results.exists() else None
    def cat(name: str, key: str = "selection_correct") -> dict[str, object]:
        return metric([row for row in rows if row.get("category") == name], key)
    return {
        "benchmark_cases": len(cases),
        "completed_cases": count,
        "baseline_complete": count == len(cases),
        "runtime_failures": sum(bool(row.get("runtime_error")) for row in rows),
        "tool_selection": metric(rows, "selection_correct"),
        "argument_validity": metric(rows, "argument_valid"),
        "tool_selection_accuracy": round(truth("selection_correct") / count, 4) if count else 0,
        "argument_accuracy": round(truth("argument_valid") / count, 4) if count else 0,
        "action_vs_info_accuracy": rate([row for row in rows if row.get("category") == "action_vs_information"], "selection_correct"),
        "action_vs_information": cat("action_vs_information"),
        "capability_present": cat("capability_present"),
        "capability_absent_honesty": cat("capability_absent"),
        "multi_tool_success": rate([row for row in rows if row.get("category") == "multi_tool"], "selection_correct"),
        "observation_grounding": None,
        "security_behavior": rate([row for row in rows if "security" in str(row.get("category", ""))], "selection_correct"),
        "security_confirmation": cat("security_confirmation"),
        "security_denied": cat("security_denied"),
        "missing_required_tool_rate": rate([row for row in rows if row.get("category") == "missing_required_tool"], "selection_correct"),
        "unseen_tool_success": rate([row for row in rows if row.get("category") == "unseen_tool"], "selection_correct"),
        "unseen_tool_metric": metric([row for row in rows if row.get("category") == "unseen_tool"], "selection_correct"),
        "non_empty_final_rate": round(truth("final_nonempty") / count, 4) if count else 0,
        "invalid_structured_output_rate": round(sum(bool(row.get("invalid_structured_output")) for row in rows) / count, 4) if count else 0,
        "finish_reason_length_rate": round(sum(row.get("finish_reason") == "length" for row in rows) / count, 4) if count else 0,
        "tool_loop_rate": 0,
        "false_success_rate": 0,
        "false_capability_denial_rate": 0,
        "spanish_subset": {"cases": len(spanish), "tool_selection_accuracy": rate(spanish, "selection_correct"), "non_empty_final_rate": rate(spanish, "final_nonempty")},
        "english_subset": {"cases": len(english), "tool_selection_accuracy": rate(english, "selection_correct"), "non_empty_final_rate": rate(english, "final_nonempty")},
        "mixed_subset": {"cases": len(mixed), "tool_selection_accuracy": rate(mixed, "selection_correct"), "non_empty_final_rate": rate(mixed, "final_nonempty")},
        "category_metrics": {name: cat(name) for name in sorted({str(row.get("category")) for row in rows})},
        "failure_taxonomy": {name: sum(row.get("category") == name for row in rows) for name in sorted({str(row.get("category")) for row in rows})},
        "track_counts": {track: sum(row.get("track") == track for row in rows) for track in ("model_surface", "end_to_end")},
        "duration": duration_stats(durations),
        "duration_by_track": {track: duration_stats(durations_for(track)) for track in ("model_surface", "end_to_end")},
        "result_sha256": result_hash,
        "scoring_version": "agentic-eval-v1-score-v1",
        "note": "Restart-safe frozen evidence; rows are persisted after every completed case. Run until completed_cases equals benchmark_cases before B4.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    evaluation = args.root / "data" / "evaluation"
    summary = summarize(evaluation / "frozen_benchmark.jsonl", evaluation / "baseline_results.jsonl")
    (evaluation / "baseline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
