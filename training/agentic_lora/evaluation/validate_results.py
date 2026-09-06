from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"case_id", "category", "track", "language", "runtime_error", "duration_ms",
            "selection_correct", "argument_valid", "finish_reason", "final_nonempty"}


def validate(benchmark: Path, results: Path) -> dict[str, object]:
    cases = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {case["case_id"]: case for case in cases}
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()] if results.exists() else []
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        if case_id not in by_id:
            errors.append(f"unknown case_id: {case_id}")
            continue
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{case_id}: missing {sorted(missing)}")
        case = by_id[case_id]
        for key in ("category", "track", "language"):
            if row.get(key) != case.get(key):
                errors.append(f"{case_id}: {key} mismatch")
        if "reasoning_content" in row or "thought" in row:
            errors.append(f"{case_id}: private reasoning field present")
    return {"valid": not errors, "benchmark_cases": len(cases), "result_rows": len(rows),
            "unique_case_ids": len(seen), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).parents[1] / "data" / "evaluation"
    parser.add_argument("--benchmark", type=Path, default=root / "frozen_benchmark.jsonl")
    parser.add_argument("--results", type=Path, default=root / "baseline_results.jsonl")
    args = parser.parse_args()
    report = validate(args.benchmark, args.results)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
