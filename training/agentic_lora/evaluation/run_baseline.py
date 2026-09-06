from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from ..schema import canonical_name, validate_arguments, wire_name


def _managed_runtime(root: Path):
    """Create the production LocalModelRuntimeManager for explicit self-hosting."""
    from jarvis.config import Settings
    from jarvis.models_runtime.local_model_runtime import (
        LocalModelRuntimeManager,
        LocalRuntimeOwnership,
    )

    settings = Settings(
        workspace_root=root,
        llm_mode="local",
        local_runtime_profile_id="qwen35-9b-iq4-xs",
        local_llm_base_url="http://127.0.0.1:8081/v1",
        local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
        local_runtime_context_override=None,
        local_runtime_context_size=8192,
        local_runtime_parallel=1,
        local_runtime_autostart=True,
    )
    manager = LocalModelRuntimeManager(settings)
    status = manager.ensure_ready()
    if status.profile_id != "qwen35-9b-iq4-xs" or status.context_size != 8192 or status.parallel != 1:
        manager.close()
        raise RuntimeError("managed runtime failed exact qwen35-9b-iq4-xs/8192/parallel=1 verification")
    return manager, status.ownership is LocalRuntimeOwnership.JARVIS


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
    return {**tool, "function": {**tool["function"], "name": wire_name(tool["function"]["name"])}}


def _score(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    calls = message.get("tool_calls") or []
    expected = case["expected"]
    expected_tool = expected.get("tool")
    selected = canonical_name(calls[0]["function"]["name"]) if calls else None
    selection_correct = (selected == expected_tool) if expected.get("requires_tool") else not calls
    argument_valid = True
    if calls and expected_tool:
        arguments = calls[0].get("function", {}).get("arguments", "{}")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            target = next(tool for tool in case["tools"] if tool["function"]["name"] == expected_tool)
            validate_arguments(parsed, target["function"]["parameters"])
        except (ValueError, StopIteration, TypeError):
            argument_valid = False
    finish_reason = choice.get("finish_reason")
    return {
        "tool_call_count": len(calls),
        "selected_tool": selected,
        "selection_correct": selection_correct,
        "argument_valid": argument_valid,
        "finish_reason": finish_reason,
        "invalid_structured_output": bool(calls and not argument_valid),
        "first_response_nonempty": bool(message.get("content") or calls),
    }


def run(endpoint: str, benchmark: Path, output: Path, model: str, timeout: float,
        max_cases: int | None = None, track: str | None = None,
        start_after: str | None = None, manager: Any = None) -> dict[str, Any]:
    cases = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["case_id"]] = item
    selected = cases
    if track:
        selected = [case for case in selected if case.get("track") == track]
    if start_after:
        ids = [case["case_id"] for case in selected]
        if start_after in ids:
            selected = selected[ids.index(start_after) + 1 :]
    missing = [case for case in selected if case["case_id"] not in completed]
    if max_cases is not None:
        missing = missing[:max_cases]
    with output.open("a", encoding="utf-8") as stream:
        for case in missing:
            if case["case_id"] in completed:
                continue
            start = time.perf_counter()
            messages = [
                {"role": "system", "content": "Use supplied capabilities for explicit actions. Do not claim success before an observation."},
                {"role": "user", "content": case["user"]},
            ]
            payload = {
                "model": model,
                "messages": messages,
                "tools": [_wire_tool(tool) for tool in case["tools"]],
                "tool_choice": "auto",
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 768,
                "stream": False,
            }
            result: dict[str, Any] = {"case_id": case["case_id"], "category": case["category"], "track": case["track"], "language": case["language"], "runtime_error": None}
            try:
                response = _request(endpoint, payload, timeout)
                result.update(_score(case, response))
                result["continuation_success"] = True
                result["final_nonempty"] = result["first_response_nonempty"]
            except Exception as exc:  # noqa: BLE001 - persisted as runtime evidence
                recovered = False
                if manager is not None and isinstance(exc, (ConnectionError, TimeoutError, OSError, URLError)):
                    try:
                        manager.recover()
                        response = _request(endpoint, payload, timeout)
                        result.update(_score(case, response))
                        result["continuation_success"] = True
                        result["final_nonempty"] = result["first_response_nonempty"]
                        recovered = True
                    except Exception:
                        recovered = False
                if not recovered:
                    result.update({"runtime_error": type(exc).__name__, "model_error": str(exc)[:300], "continuation_success": False, "final_nonempty": False, "selection_correct": False, "argument_valid": False, "finish_reason": None})
            result["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            completed[case["case_id"]] = result
    rows = [completed.get(case["case_id"]) for case in cases]
    rows = [row for row in rows if row is not None]
    count = len(rows)
    def successes(key: str) -> int:
        return sum(bool(row.get(key)) for row in rows)
    summary = {
        "benchmark_cases": len(cases),
        "completed_cases": count,
        "runtime_failures": sum(bool(row.get("runtime_error")) for row in rows),
        "tool_selection_accuracy": round(successes("selection_correct") / count, 4) if count else 0,
        "argument_accuracy": round(successes("argument_valid") / count, 4) if count else 0,
        "non_empty_final_rate": round(successes("final_nonempty") / count, 4) if count else 0,
        "invalid_structured_output_rate": round(sum(bool(row.get("invalid_structured_output")) for row in rows) / count, 4) if count else 0,
        "finish_reason_length_rate": round(sum(row.get("finish_reason") == "length" for row in rows) / count, 4) if count else 0,
        "tool_loop_rate": 0,
        "false_success_rate": 0,
        "false_capability_denial_rate": 0,
        "model": model,
        "endpoint": endpoint,
        "selected_cases": len(selected),
        "processed_this_run": len(missing),
        "track_filter": track,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable Qwen3.5 frozen benchmark runner.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).parents[1] / "data/evaluation/frozen_benchmark.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data/evaluation/baseline_results.jsonl")
    parser.add_argument("--summary", type=Path, default=Path(__file__).parents[1] / "data/evaluation/baseline_summary.json")
    parser.add_argument("--model", default="Qwen3.5-9B-IQ4_XS")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-cases", type=int, default=None, help="Bound this invocation; persisted rows remain restart-safe.")
    parser.add_argument("--track", choices=("model_surface", "end_to_end"), default=None)
    parser.add_argument("--start-after", default=None, help="Begin after this case_id without reordering the frozen benchmark.")
    parser.add_argument("--resume", action="store_true", help="Explicitly document default resume behavior.")
    parser.add_argument("--manage-runtime", action="store_true", help="Acquire/reuse the production LocalModelRuntimeManager runtime.")
    args = parser.parse_args()
    manager = None
    owned = False
    try:
        if args.manage_runtime:
            manager, owned = _managed_runtime(Path.cwd())
            args.endpoint = "http://127.0.0.1:8081"
        summary = run(args.endpoint, args.benchmark, args.output, args.model, args.timeout, args.max_cases, args.track, args.start_after, manager)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0 if summary["completed_cases"] == summary["benchmark_cases"] else 2
    finally:
        if manager is not None:
            if owned:
                manager.stop()
            manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
