"""Measure frozen PRE-LORA-3.1 direct retrieval without modifying case files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from jarvis.actions.discovery import DISCOVERY_ACTION
from jarvis.bootstrap import build_application
from jarvis.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", choices=("development", "blind_final"))
    args = parser.parse_args()
    base = Path("docs/evaluations")
    case_path = base / f"prelora31_v1_{args.partition}.json"
    payload = case_path.read_bytes()
    manifest = json.loads((base / "prelora31_v1_manifest.json").read_text(encoding="utf-8"))
    expected_hash = manifest["partitions"][args.partition]["sha256"]
    assert hashlib.sha256(payload).hexdigest() == expected_hash
    cases = json.loads(payload)
    app = build_application(Settings(_env_file=None))
    rows = []
    for case in cases:
        names = [
            item["function"]["name"]
            for item in app.agent_controller.available_tool_schemas(case["prompt"])
            if item["function"]["name"] != DISCOVERY_ACTION
        ]
        required = set(case["required"])
        selected = set(names)
        required_families = {name.split(".", 1)[0] for name in required}
        selected_families = {name.split(".", 1)[0] for name in selected}
        rows.append({
            "id": case["id"], "intent": case["intent"],
            "selected": names, "required": case["required"],
            "exact": selected == required,
            "recall": len(selected & required) / len(required) if required else 1.0,
            "precision": len(selected & required) / len(selected) if selected else 1.0,
            "family_recall": len(selected_families & required_families) / len(required_families) if required_families else 1.0,
            "operation_recall": len(selected & required) / len(required) if required else 1.0,
            "discovery_needed": not required <= selected,
        })
    actionable = [row for row in rows if row["intent"] != "information"]
    summary = {
        "benchmark": "prelora31-v1", "partition": args.partition,
        "sha256": expected_hash, "cases": len(cases), "actionable_cases": len(actionable),
        "required_tool_recall": sum(row["recall"] for row in actionable) / len(actionable),
        "retrieval_precision": sum(row["precision"] for row in actionable) / len(actionable),
        "exact_tool_set": sum(row["exact"] for row in actionable) / len(actionable),
        "family_recall": sum(row["family_recall"] for row in actionable) / len(actionable),
        "direct_hit_rate": sum(not row["discovery_needed"] for row in actionable) / len(actionable),
        "mean_selected_count": sum(len(row["selected"]) for row in actionable) / len(actionable),
    }
    result = {"summary": summary, "cases": rows}
    output = base / f"prelora31_v1_{args.partition}_retrieval.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
