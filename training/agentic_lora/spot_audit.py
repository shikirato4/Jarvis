from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .schema import validate_record


def main() -> int:
    root = Path(__file__).parent
    files = root / "data" / "canonical"
    records = [
        json.loads(line)
        for name in ("train.jsonl", "validation.jsonl", "heldout.jsonl", "hard_regression.jsonl")
        for line in (files / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        if record["split"] == "heldout" and len(selected) < 10:
            selected.append(record)
    for kind in ("control", "real", "fictitious"):
        candidate = next((record for record in records if record["real_or_fictitious"] == kind), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    for record in records:
        if len(selected) >= 60:
            break
        key = (record["locale"], record["real_or_fictitious"], record["category_tags"][0])
        if key not in seen:
            selected.append(record)
            seen.add(key)
    errors: list[str] = []
    for record in selected:
        try:
            validate_record(record)
        except ValueError as exc:
            errors.append(f"{record['record_id']}: {exc}")
    report = {
        "audit_version": "spot-audit-v1",
        "records_audited": len(selected),
        "selection": {"locales": dict(Counter(item["locale"] for item in selected)), "content": dict(Counter(item["real_or_fictitious"] for item in selected)), "categories": dict(Counter(item["category_tags"][0] for item in selected))},
        "errors": errors,
        "systematic_issues": [],
        "verdict": "PASS" if not errors else "FAIL",
    }
    out = root / "data" / "spot_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
