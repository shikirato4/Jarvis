from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from . import SCHEMA_VERSION
from .schema import stable_json, validate_dataset, wire_name

ASSISTANT_HEADER = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"


def _wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in messages:
        message = dict(original)
        if message.get("role") == "tool":
            message["content"] = stable_json(message["content"])
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {
                    **call,
                    "function": {
                        **call["function"],
                        "name": wire_name(call["function"]["name"]),
                    },
                }
                for call in message["tool_calls"]
            ]
        result.append(message)
    return result


def _wire_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **tool,
            "function": {**tool["function"], "name": wire_name(tool["function"]["name"])},
        }
        for tool in tools
    ]


def _assistant_spans(rendered: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(re.escape(ASSISTANT_HEADER), rendered):
        start = match.end()
        end = rendered.find(END_MARKER, start)
        if end < 0:
            raise ValueError("assistant block has no end marker")
        spans.append((start, end))
    return spans


def compile_record(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    messages = _wire_messages(record["messages"])
    tools = _wire_tools(record["tools"])
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    spans = _assistant_spans(rendered)
    labels = [-100] * len(input_ids)
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            labels[index] = input_ids[index]
    if not any(label != -100 for label in labels):
        raise ValueError(f"{record['record_id']}: no assistant labels")
    schema_text = stable_json(tools)
    return {
        "record_id": record["record_id"],
        "schema_version": SCHEMA_VERSION,
        "compiler_version": "compiler-v1",
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "visible_token_count": len(input_ids),
        "tool_schema_token_count": len(tokenizer(schema_text, add_special_tokens=False)["input_ids"]),
        "assistant_label_token_count": sum(label != -100 for label in labels),
        "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def compile_split(source_path: Path, output_path: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    records = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    compiled = [compile_record(record, tokenizer) for record in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in compiled),
        encoding="utf-8",
    )
    lengths = sorted(item["visible_token_count"] for item in compiled)
    def percentile(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(round((len(lengths) - 1) * p)))]
    return compiled, {
        "count": len(compiled),
        "average": round(sum(lengths) / len(lengths), 2),
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": max(lengths),
        "over_1024": sum(length > 1024 for length in lengths),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile canonical B2 records with the pinned Qwen tokenizer.")
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--tokenizer", type=Path, default=Path(__file__).parent / "base_metadata")
    args = parser.parse_args()
    canonical = args.root / "data" / "canonical"
    compiled_root = args.root / "data" / "compiled"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    production_template = args.tokenizer / "production_chat_template.jinja"
    if production_template.exists():
        tokenizer.chat_template = production_template.read_text(encoding="utf-8")
    source_records = []
    for filename in ("train.jsonl", "validation.jsonl", "heldout.jsonl", "hard_regression.jsonl"):
        source_records.extend(json.loads(line) for line in (canonical / filename).read_text(encoding="utf-8").splitlines() if line.strip())
    errors = validate_dataset(source_records)
    if errors:
        raise SystemExit("canonical validation failed: " + "; ".join(errors[:5]))
    stats: dict[str, Any] = {}
    for split in ("train", "validation", "heldout", "hard_regression"):
        _, split_stats = compile_split(canonical / f"{split}.jsonl", compiled_root / f"{split}.jsonl", tokenizer)
        stats[split] = split_stats
    all_lengths = [json.loads(line)["visible_token_count"] for split in stats for line in (compiled_root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    report = {
        "schema_version": SCHEMA_VERSION,
        "compiler_version": "compiler-v1",
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "chat_template_source": "production_chat_template.jinja",
        "sequence_limit": 1024,
        "records": len(all_lengths),
        "average": round(sum(all_lengths) / len(all_lengths), 2),
        "median": sorted(all_lengths)[len(all_lengths) // 2],
        "p90": sorted(all_lengths)[int(round((len(all_lengths) - 1) * 0.9))],
        "p95": sorted(all_lengths)[int(round((len(all_lengths) - 1) * 0.95))],
        "p99": sorted(all_lengths)[int(round((len(all_lengths) - 1) * 0.99))],
        "maximum": max(all_lengths),
        "over_1024": sum(length > 1024 for length in all_lengths),
        "by_split": stats,
    }
    (compiled_root / "tokenization_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
