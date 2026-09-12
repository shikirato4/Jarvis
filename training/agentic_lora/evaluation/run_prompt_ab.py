from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy

from .run_baseline import _managed_runtime
from ..schema import canonical_name, validate_arguments, wire_name


CONTROL_PROMPT = (
    "You are Jarvis. Use supplied capabilities for explicit actions. Informational "
    "questions do not require tools. Respect SecurityGate, inspect observations, "
    "and never claim success before an observation."
)

TARGET_CATEGORIES = (
    "correct_tool_selection",
    "argument_extraction",
    "action_vs_information",
    "capability_present",
    "capability_absent",
    "irrelevant_distractors",
    "parameter_ambiguity",
    "multi_tool",
    "tool_failure",
    "no_false_success",
    "no_false_capability_denial",
    "output_concision",
)


def _request(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wire_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        **tool,
        "function": {
            **tool["function"],
            "name": wire_name(tool["function"]["name"]),
        },
    }


def _cases(source: Path, per_category: int) -> list[dict[str, Any]]:
    surfaces = ("real_tool", "fictitious_tool", "non_tool_control")
    counts = {
        (category, surface): 0
        for category in TARGET_CATEGORIES
        for surface in surfaces
    }
    selected: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tags = record.get("category_tags", [])
        surface = next((item for item in surfaces if item in tags), None)
        category = next(
            (
                item
                for item in tags
                if (item, surface) in counts
                and counts[(item, surface)] < per_category
            ),
            None,
        )
        if category is None or surface is None:
            continue
        user = next(item for item in record["messages"] if item["role"] == "user")
        assistant = next(
            item for item in record["messages"] if item["role"] == "assistant"
        )
        expected_calls = assistant.get("tool_calls") or []
        expected_tool = (
            expected_calls[0]["function"]["name"] if expected_calls else None
        )
        selected.append(
            {
                "case_id": record["record_id"],
                "category": category,
                "surface": surface,
                "user": user["content"],
                "tools": record.get("tools") or [],
                "expected_tool": expected_tool,
                "requires_tool": bool(
                    record.get("expected_behavior", {}).get("requires_tool")
                ),
                "expected_messages": record["messages"],
            }
        )
        counts[(category, surface)] += 1
        if all(value >= per_category for value in counts.values()):
            break
    return selected


def _prompt(arm: str, tools: list[dict[str, Any]]) -> str:
    if arm == "control":
        return CONTROL_PROMPT
    policy = AgenticPromptPolicy()
    dynamic = policy.capability_context(tools)
    return f"{policy.static_prompt}\n\n{dynamic}"


def _score_first(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    selected = (
        canonical_name(calls[0].get("function", {}).get("name", ""))
        if calls
        else None
    )
    selection_correct = (
        selected == case["expected_tool"]
        if case["requires_tool"]
        else not calls
    )
    argument_valid = not calls
    if calls:
        try:
            arguments = calls[0]["function"].get("arguments", "{}")
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            target = next(
                tool
                for tool in case["tools"]
                if tool["function"]["name"] == selected
            )
            validate_arguments(parsed, target["function"]["parameters"])
            argument_valid = True
        except (KeyError, StopIteration, TypeError, ValueError):
            argument_valid = False
    return {
        "message": message,
        "tool_call_count": len(calls),
        "selected_tool": selected,
        "selection_correct": selection_correct,
        "argument_valid": argument_valid,
        "unnecessary_tool": bool(calls and not case["requires_tool"]),
        "first_response_nonempty": bool(message.get("content") or calls),
        "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
    }


def run(
    *,
    endpoint: str,
    source: Path,
    output: Path,
    model: str,
    timeout: float,
    per_category: int,
) -> dict[str, Any]:
    cases = _cases(source, per_category)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for arm in ("control", "candidate"):
        for case in cases:
            started = time.perf_counter()
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": _prompt(arm, case["tools"])},
                    {"role": "user", "content": case["user"]},
                ],
                "tools": [_wire_tool(tool) for tool in case["tools"]],
                "tool_choice": "auto",
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "max_tokens": 512,
                "stream": False,
            }
            row = {
                "arm": arm,
                "case_id": case["case_id"],
                "category": case["category"],
                "surface": case["surface"],
                "runtime_error": None,
            }
            try:
                response = _request(endpoint, payload, timeout)
                scored = _score_first(case, response)
                row.update({key: value for key, value in scored.items() if key != "message"})
            except Exception as exc:  # noqa: BLE001 - persisted evaluation evidence
                row.update(
                    {
                        "runtime_error": type(exc).__name__,
                        "selection_correct": False,
                        "argument_valid": False,
                        "unnecessary_tool": False,
                        "first_response_nonempty": False,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                    }
                )
            row["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
            rows.append(row)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def summarize(arm: str) -> dict[str, Any]:
        group = [row for row in rows if row["arm"] == arm]
        count = len(group)
        rate = lambda key: round(sum(bool(row.get(key)) for row in group) / count, 4) if count else 0  # noqa: E731
        prompt_tokens = [row["prompt_tokens"] for row in group if isinstance(row.get("prompt_tokens"), int)]
        return {
            "cases": count,
            "runtime_failures": sum(bool(row["runtime_error"]) for row in group),
            "tool_selection_accuracy": rate("selection_correct"),
            "argument_validity_rate": rate("argument_valid"),
            "unnecessary_tool_rate": rate("unnecessary_tool"),
            "first_response_nonempty_rate": rate("first_response_nonempty"),
            "mean_latency_ms": round(statistics.mean(row["duration_ms"] for row in group), 2) if group else 0,
            "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 2) if prompt_tokens else None,
            "by_category": {
                category: {
                    "count": len(items),
                    "selection_accuracy": round(sum(bool(item.get("selection_correct")) for item in items) / len(items), 4),
                }
                for category in TARGET_CATEGORIES
                if (items := [row for row in group if row["category"] == category])
            },
            "by_surface": {
                surface: {
                    "count": len(items),
                    "selection_accuracy": round(sum(bool(item.get("selection_correct")) for item in items) / len(items), 4),
                }
                for surface in (
                    "real_tool",
                    "fictitious_tool",
                    "non_tool_control",
                )
                if (items := [row for row in group if row["surface"] == surface])
            },
        }

    control = summarize("control")
    candidate = summarize("candidate")
    return {
        "evaluation_source": str(source),
        "deterministic": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "max_tokens": 512,
        },
        "control": control,
        "candidate": candidate,
        "delta": {
            key: round(candidate[key] - control[key], 4)
            for key in (
                "tool_selection_accuracy",
                "argument_validity_rate",
                "unnecessary_tool_rate",
                "first_response_nonempty_rate",
                "mean_latency_ms",
            )
        },
        "sample_size_per_arm": len(cases),
        "model": model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-only PRE-LORA control/candidate evaluation")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).parents[1] / "data/canonical/validation.jsonl",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data/prelora_prompt_ab_results.jsonl")
    parser.add_argument("--summary", type=Path, default=Path(__file__).parents[1] / "data/prelora_prompt_ab_summary.json")
    parser.add_argument("--model", default="Qwen3.5-9B-IQ4_XS")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--per-category", type=int, default=1)
    parser.add_argument(
        "--manage-runtime",
        action="store_true",
        help="Acquire the exact production local runtime for this evaluation.",
    )
    args = parser.parse_args()
    manager = None
    owned = False
    try:
        if args.manage_runtime:
            manager, owned = _managed_runtime(Path.cwd())
            args.endpoint = "http://127.0.0.1:8081"
        summary = run(
            endpoint=args.endpoint,
            source=args.source,
            output=args.output,
            model=args.model,
            timeout=args.timeout,
            per_category=args.per_category,
        )
        args.summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        failures = (
            summary["control"]["runtime_failures"]
            or summary["candidate"]["runtime_failures"]
        )
        return 0 if not failures else 2
    finally:
        if manager is not None:
            if owned:
                manager.stop()
            manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
