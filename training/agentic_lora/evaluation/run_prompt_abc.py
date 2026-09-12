from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy
from jarvis.models_runtime.base import ModelMessage

from ..schema import canonical_name, validate_arguments, wire_name
from .prelora2_dev_cases import CASES
from .run_baseline import _managed_runtime
from .run_prompt_ab import _request, _wire_tool


ARMS = ("control", "compact", "full")


def _policy(arm: str) -> AgenticPromptPolicy:
    return AgenticPromptPolicy(profile={"control": "off", "compact": "compact", "full": "full"}[arm])


def _system_prompt(arm: str, tools: list[dict[str, Any]]) -> str:
    messages = _policy(arm).compose(
        [ModelMessage(role="system", content="You are Jarvis.")],
        tools,
        all_tools=tools,
    )
    return messages[0].content


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    value = call.get("function", {}).get("arguments", {})
    return json.loads(value) if isinstance(value, str) else value


def _false_success(content: str, failed: bool) -> bool:
    if not failed:
        return False
    folded = content.casefold()
    success = any(word in folded for word in ("done", "completed", "successfully", "hecho", "listo", "completado", "cambié"))
    negated = any(word in folded for word in ("not ", "no ", "failed", "falló", "denied", "deneg"))
    return success and not negated


def _run_case(
    endpoint: str,
    model: str,
    timeout: float,
    arm: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(arm, case["tools"])},
        {"role": "user", "content": case["user"]},
    ]
    expected = list(case["sequence"])
    calls: list[str] = []
    arguments_valid = True
    final = ""
    prompt_tokens = 0
    completion_tokens = 0
    failed_observed = False
    started = time.perf_counter()
    for step in range(max(len(expected) + 1, 2)):
        response = _request(
            endpoint,
            {
                "model": model,
                "messages": messages,
                "tools": [_wire_tool(tool) for tool in case["tools"]],
                "tool_choice": "auto",
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "max_tokens": 512,
                "stream": False,
            },
            timeout,
        )
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        message = (response.get("choices") or [{}])[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            final = str(message.get("content") or "")
            break
        call = tool_calls[0]
        selected = canonical_name(str(call.get("function", {}).get("name") or ""))
        calls.append(selected)
        try:
            schema = next(tool for tool in case["tools"] if tool["function"]["name"] == selected)
            validate_arguments(_arguments(call), schema["function"]["parameters"])
        except (KeyError, StopIteration, TypeError, ValueError):
            arguments_valid = False
        if case["observations"]:
            observation_index = min(step, len(case["observations"]) - 1)
            observation = case["observations"][observation_index]
        else:
            observation = {
                "status": "failed",
                "error_type": "UnexpectedToolCall",
                "data": {},
            }
        failed_observed = failed_observed or observation.get("status") == "failed"
        messages.append(message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": wire_name(selected),
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )
    return {
        "arm": arm,
        "case_id": case["id"],
        "category": case["category"],
        "expected_sequence": expected,
        "actual_sequence": calls,
        "tool_selection": calls == expected,
        "first_tool_correct": (calls[:1] == expected[:1]) if expected else not calls,
        "argument_valid": arguments_valid,
        "unnecessary_tool": not expected and bool(calls),
        "multi_tool_continuation": len(expected) < 2 or calls == expected,
        "failure_recovery": case["category"] != "failure_recovery" or calls == expected,
        "false_execution": _false_success(final, failed_observed),
        "final_nonempty": bool(final.strip()),
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "ttft_ms": None,
    }


def run(endpoint: str, model: str, timeout: float, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for case in CASES:
            try:
                rows.append(_run_case(endpoint, model, timeout, arm, case))
            except Exception as exc:  # noqa: BLE001 - evaluation evidence
                rows.append(
                    {
                        "arm": arm,
                        "case_id": case["id"],
                        "category": case["category"],
                        "runtime_error": type(exc).__name__,
                        "tool_selection": False,
                        "argument_valid": False,
                        "final_nonempty": False,
                    }
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def summarize(arm: str) -> dict[str, Any]:
        group = [row for row in rows if row["arm"] == arm]
        rate = lambda key: round(sum(bool(row.get(key)) for row in group) / len(group), 4)  # noqa: E731
        latency = [row["total_latency_ms"] for row in group if "total_latency_ms" in row]
        tokens = [row["prompt_tokens_total"] for row in group if "prompt_tokens_total" in row]
        return {
            "cases": len(group),
            "runtime_failures": sum(bool(row.get("runtime_error")) for row in group),
            "tool_selection_accuracy": rate("tool_selection"),
            "first_tool_accuracy": rate("first_tool_correct"),
            "argument_validity": rate("argument_valid"),
            "unnecessary_tool_rate": rate("unnecessary_tool"),
            "multi_tool_continuation": rate("multi_tool_continuation"),
            "failure_recovery": rate("failure_recovery"),
            "false_execution_rate": rate("false_execution"),
            "final_nonempty_rate": rate("final_nonempty"),
            "mean_total_latency_ms": round(statistics.mean(latency), 2) if latency else None,
            "mean_cumulative_prompt_tokens": round(statistics.mean(tokens), 2) if tokens else None,
            "by_category": {
                category: {
                    "count": len(items),
                    "tool_selection": round(sum(bool(item.get("tool_selection")) for item in items) / len(items), 4),
                }
                for category in sorted({row["category"] for row in group})
                if (items := [row for row in group if row["category"] == category])
            },
        }

    return {
        "source": "prelora2_dev_cases.py",
        "deterministic": {"temperature": 0, "top_p": 1, "seed": 42, "max_tokens": 512},
        "model": model,
        "sample_size_per_arm": len(CASES),
        **{arm: summarize(arm) for arm in ARMS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PRE-LORA off/compact/full capability evaluation")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--model", default="Qwen3.5-9B-IQ4_XS")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--manage-runtime", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data/prelora2_prompt_abc_results.jsonl")
    parser.add_argument("--summary", type=Path, default=Path(__file__).parents[1] / "data/prelora2_prompt_abc_summary.json")
    args = parser.parse_args()
    manager = None
    owned = False
    try:
        if args.manage_runtime:
            manager, owned = _managed_runtime(Path.cwd())
        summary = run(args.endpoint, args.model, args.timeout, args.output)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if manager is not None:
            if owned:
                manager.stop()
            manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
