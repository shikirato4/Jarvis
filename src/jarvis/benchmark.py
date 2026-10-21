from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable
from uuid import uuid4

from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.hybrid import RequestExecutionContext


def build_application(settings: Settings):
    """Lazy compatibility seam used by benchmarks and their test harnesses."""
    from jarvis.bootstrap import build_application as build

    return build(settings)


def _measure(function: Callable[[], Any], *, iterations: int = 3) -> dict[str, Any]:
    samples: list[float] = []
    last_value: Any = None
    for _ in range(max(iterations, 1)):
        started = time.perf_counter()
        last_value = function()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "iterations": len(samples),
        "last_value": last_value,
    }


def run_benchmark(
    include_search: bool = False,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run local microbenchmarks; no model throughput is inferred or fabricated."""

    del root  # Kept for the code-agent CLI compatibility surface.
    routing = _measure(lambda: "auto", iterations=1_000)
    retrieval = _measure(
        lambda: sum(index for index in range(500)),
        iterations=50,
    )
    result: dict[str, Any] = {
        "benchmark_type": "local_microbenchmark",
        "simulated": True,
        "measurements": {
            "routing_overhead": routing,
            "retrieval_cpu_baseline": retrieval,
        },
        "model_generation": {
            "measured": False,
            "tokens_per_second": None,
            "reason": "No provider request is executed by the local microbenchmark.",
        },
        "partitions": [
            "startup",
            "routing",
            "health",
            "ttft",
            "generation",
            "tool",
            "retrieval",
            "multimodal",
            "tts",
        ],
    }
    if include_search:
        result["measurements"]["search_stub"] = _measure(
            lambda: "needle" in "local deterministic search needle",
            iterations=100,
        )
    return result


def run_real_benchmark(
    *,
    root: str | Path | None = None,
    prompt: str = "hola jarvis",
    mode: str | None = None,
    breakdown: bool = False,
) -> dict[str, Any]:
    del root
    started = time.perf_counter()
    settings = Settings(llm_mode=mode or "auto")
    application = build_application(settings)
    build_ms = (time.perf_counter() - started) * 1000
    start_started = time.perf_counter()
    application.start()
    start_ms = (time.perf_counter() - start_started) * 1000
    request_id = f"benchmark-{uuid4().hex[:12]}"
    request = _request(prompt, request_id=request_id, stream=False)
    inference_started = time.perf_counter()
    try:
        response = application.model_service.infer(
            request,
            execution_context=RequestExecutionContext(request_id=request_id),
        )
        generation_ms = (time.perf_counter() - inference_started) * 1000
        content_chars = len(response.content)
        result = {
            "status": "ok",
            "benchmark_type": "live_provider_request",
            "simulated": False,
            "selected_mode": application.model_service.provider_selection().selected_mode.value,
            "active_provider": response.provider_name,
            "model": response.model_name,
            "timings": {
                "application_build_ms": round(build_ms, 2),
                "service_start_ms": round(start_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "ttft_ms": None,
            "tokens_per_second": None,
            "content_chars": content_chars,
            "note": "Synchronous inference does not expose TTFT or verified token counts.",
        }
        if breakdown:
            result["provider_metadata"] = {
                key: value
                for key, value in response.metadata.items()
                if key
                in {
                    "selected_mode",
                    "active_provider",
                    "request_provider",
                    "active_model",
                    "provider_state",
                    "fallback_reason",
                    "request_id",
                }
            }
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unavailable",
            "benchmark_type": "live_provider_request",
            "simulated": False,
            "selected_mode": application.model_service.provider_selection().selected_mode.value,
            "timings": {
                "application_build_ms": round(build_ms, 2),
                "service_start_ms": round(start_ms, 2),
                "failure_ms": round((time.perf_counter() - inference_started) * 1000, 2),
            },
            "error_type": type(exc).__name__,
            "tokens_per_second": None,
        }
    finally:
        application.stop()


def run_streaming_benchmark(
    *,
    root: str | Path | None = None,
    prompt: str = "hola jarvis",
    mode: str | None = None,
    debug_stream: bool = False,
) -> dict[str, Any]:
    del root
    application = build_application(Settings(llm_mode=mode or "auto"))
    application.start()
    request_id = f"benchmark-stream-{uuid4().hex[:12]}"
    started = time.perf_counter()
    first_token_ms: float | None = None
    generation_chars = 0
    chunks = 0
    terminal_error: str | None = None
    terminal_metadata: dict[str, Any] = {}
    try:
        for chunk in application.model_service.stream(
            _request(prompt, request_id=request_id, stream=True),
            execution_context=RequestExecutionContext(request_id=request_id),
        ):
            if chunk.text:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                generation_chars += len(chunk.text)
                chunks += 1
            if chunk.done:
                terminal_error = chunk.error
                terminal_metadata = dict(chunk.metadata)
        total_ms = (time.perf_counter() - started) * 1000
        result = {
            "status": "ok" if not terminal_error else "unavailable",
            "benchmark_type": "live_streaming_provider_request",
            "simulated": False,
            "selected_mode": application.model_service.provider_selection().selected_mode.value,
            "active_provider": terminal_metadata.get("active_provider"),
            "model": terminal_metadata.get("active_model"),
            "ttft_ms": round(first_token_ms, 2) if first_token_ms is not None else None,
            "generation_ms": (
                round(total_ms - first_token_ms, 2)
                if first_token_ms is not None
                else None
            ),
            "total_ms": round(total_ms, 2),
            "chunks": chunks,
            "content_chars": generation_chars,
            "tokens_per_second": None,
            "error": terminal_error,
            "note": "Throughput is omitted because the provider stream does not expose verified token counts.",
        }
        if debug_stream:
            result["safe_terminal_metadata"] = {
                key: value
                for key, value in terminal_metadata.items()
                if key
                in {
                    "selected_mode",
                    "active_provider",
                    "request_provider",
                    "active_model",
                    "provider_state",
                    "fallback_reason",
                    "request_id",
                }
            }
        return result
    finally:
        application.stop()


def _request(prompt: str, *, request_id: str, stream: bool) -> ModelRequest:
    return ModelRequest(
        prompt=prompt,
        messages=[ModelMessage(role="user", content=prompt)],
        logical_model="general_assistant",
        task_type="assistant",
        required_capabilities=("chat",),
        correlation_id=request_id,
        stream=stream,
    )


def format_benchmark(result: dict[str, Any]) -> str:
    return _format(result)


def format_real_benchmark(result: dict[str, Any]) -> str:
    return _format(result)


def format_streaming_benchmark(result: dict[str, Any]) -> str:
    return _format(result)


def _format(result: dict[str, Any]) -> str:
    lines = [
        "Jarvis Benchmark",
        f"Benchmark: {result.get('benchmark_type', 'unknown')}",
        f"Status: {result.get('status', 'ok')}",
        f"Simulated: {str(bool(result.get('simulated'))).lower()}",
        "OpenAI: blocked",
        "Gemini: blocked",
    ]
    timings = result.get("timings")
    if isinstance(timings, dict):
        lines.extend(f"{key}: {value}" for key, value in timings.items())
    for key in ("ttft_ms", "generation_ms", "total_ms", "tokens_per_second"):
        if key in result:
            lines.append(f"{key}: {result.get(key)}")
    if result.get("note"):
        lines.append(str(result["note"]))
    return "\n".join(lines)
