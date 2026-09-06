from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generator import write_dataset
from .schema import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic JARVIS Agentic LoRA pilot.")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    records, report = write_dataset(args.output, args.seed)
    errors = validate_dataset(records, split_counts={"train": 800, "validation": 120, "heldout": 200, "hard_regression": 80})
    report["validation_failures"] = errors
    report_path = args.output / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [
        "# JARVIS Agentic LoRA pilot dataset report",
        "",
        f"- Schema: {report['schema_version']}",
        f"- Generator: {report['generator_version']}",
        f"- Seed: {report['seed']}",
        f"- Total: {report['total_records']}",
        f"- Splits: {report['records_by_split']}",
        f"- Languages: {report['records_by_language']}",
        f"- Real/fictitious/control: {report['real_fictitious_control']}",
        f"- Categories: {report['records_by_category']}",
        f"- Estimated length tokens: average {report['estimated_average_tokens']}, maximum {report['estimated_max_tokens']}",
        f"- Held-out families: {', '.join(report['heldout_tool_families'])}",
        f"- Near-duplicate warnings: {report['near_duplicate_warnings']}",
        f"- Validation failures: {len(errors)}",
        "",
    ]
    (args.output / "dataset_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"records": len(records), "validation_failures": len(errors), "report": str(report_path)}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
