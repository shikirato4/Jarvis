from __future__ import annotations

import contextlib
import io
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from jarvis.bootstrap import build_application
from jarvis.chat_optimization import estimated_tokens_from_chars, select_chat_generation_profile, streaming_timeout_seconds
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.config import Settings
from jarvis.environment import detect_environment
from jarvis.identity import jarvis_identity_prompt, sanitize_assistant_identity
from jarvis.models_runtime.base import ModelRequest, ModelResponse, ProviderKind, StreamChunk
from jarvis.ollama_diagnostics import classify_local_model_error, local_model_failure_message
from jarvis.web_search import build_grounded_web_prompt, build_web_search_provider, should_use_web_search
from jarvis.web_search.sanitizer import sanitize_web_query


def run_benchmark(*, root: str | Path | None = None, include_search: bool = False) -> dict[str, Any]:
    project_root = Path(root).resolve(strict=False) if root else Path.cwd()
    results: dict[str, Any] = {
        "title": "Jarvis Benchmark",
        "environment": {},
        "llm": {},
        "web": {},
        "context": {},
        "code_agent": {},
        "policy": {"openai": "blocked", "gemini": "blocked", "online_llm": "disabled", "online_search": "Brave + local Ollama"},
    }

    env, doctor_ms = _measure(lambda: detect_environment())
    service, service_ms = _measure(lambda: CodeAgentRuntimeService(project_root))
    web_provider = build_web_search_provider()
    web_status, web_status_ms = _measure(web_provider.status)

    results["environment"] = {
        "doctor_ms": doctor_ms,
        "local_provider": env.recommended_local_provider or "none",
        "local_model": env.recommended_local_model or "none",
        "ollama_available": env.ollama.available,
        "brave_key_configured": web_status.configured,
    }
    results["web"]["status_ms"] = web_status_ms
    results["web"]["provider"] = web_status.provider
    results["web"]["available"] = web_status.available
    results["web"]["configured"] = web_status.configured
    if include_search and web_status.available:
        search, search_ms = _measure(lambda: web_provider.search("noticias de tecnologia hoy", max_results=web_status.max_results))
        results["web"]["search_ms"] = search_ms
        results["web"]["results"] = len(search.hits)
    else:
        results["web"]["search"] = "skipped, Brave Search API key not configured" if not web_status.available else "skipped"

    results["code_agent"]["init_ms"] = service_ms
    for mode in ("offline", "auto", "online", "disabled"):
        route, elapsed = _measure(lambda current=mode: service.llm_route("hola jarvis", llm_mode=current, allow_online=(current == "online")))
        results["llm"][mode] = {"ms": elapsed, "allowed": route.get("allowed"), "provider": route.get("provider_name"), "reason": route.get("reason")}
    memory, memory_ms = _measure(lambda: service.memory_summary(max_chars=1200))
    git_status, git_ms = _measure(service.git_summary)
    patch_list, patch_ms = _measure(lambda: service.patch_list(limit=10))
    learning, learning_ms = _measure(lambda: service.learn_stats())
    results["context"]["memory_summary_ms"] = memory_ms
    results["context"]["memory_chars"] = len(memory)
    results["code_agent"]["git_status_ms"] = git_ms
    results["code_agent"]["patch_list_ms"] = patch_ms
    results["code_agent"]["patch_count"] = len(patch_list.get("patches", [])) if isinstance(patch_list, dict) else 0
    results["code_agent"]["github_learning_ms"] = learning_ms
    results["code_agent"]["github_learning_entries"] = learning.get("entry_count", 0) if isinstance(learning, dict) else 0
    return results


def format_benchmark(result: dict[str, Any]) -> str:
    env = result.get("environment", {})
    web = result.get("web", {})
    llm = result.get("llm", {})
    context = result.get("context", {})
    code = result.get("code_agent", {})
    lines = [
        "Jarvis Benchmark",
        "",
        "Environment:",
        f"Local provider: {env.get('local_provider')}",
        f"Local model: {env.get('local_model')}",
        f"Web provider: {web.get('provider')}",
        f"Brave key: configured {str(env.get('brave_key_configured')).lower()}",
        "OpenAI: blocked",
        "Gemini: blocked",
        "",
        "LLM:",
    ]
    for mode in ("offline", "auto", "online", "disabled"):
        item = llm.get(mode, {})
        lines.append(f"{mode}: {item.get('ms')} ms, provider={item.get('provider')}, allowed={item.get('allowed')}")
    lines.extend(
        [
            "",
            "Web:",
            f"status: {web.get('status_ms')} ms",
            f"search: {web.get('search_ms', web.get('search'))}",
            f"results: {web.get('results', 0)}",
            "",
            "Context:",
            f"memory summary: {context.get('memory_summary_ms')} ms",
            f"memory chars: {context.get('memory_chars')}",
            "",
            "Code Agent:",
            f"patch list: {code.get('patch_list_ms')} ms",
            f"git status: {code.get('git_status_ms')} ms",
            f"GitHub learning: {code.get('github_learning_ms')} ms, entries={code.get('github_learning_entries')}",
        ]
    )
    return "\n".join(lines)


def run_real_benchmark(
    *,
    root: str | Path | None = None,
    prompt: str = "hola jarvis",
    mode: str | None = None,
    breakdown: bool = False,
    model_runner: Callable[[ModelRequest], ModelResponse] | None = None,
    web_provider: Any | None = None,
) -> dict[str, Any]:
    selected_modes = [mode.strip().casefold()] if mode else ["offline", "auto", "online", "disabled"]
    project_root = Path(root).resolve(strict=False) if root else Path.cwd()
    provider = web_provider or build_web_search_provider()
    web_status = provider.status()
    settings = Settings()
    result: dict[str, Any] = {
        "title": "Jarvis Real Benchmark",
        "prompt": prompt,
        "environment": {
            "root": str(project_root),
            "brave_key_configured": web_status.configured,
            "openai": "blocked",
            "gemini": "blocked",
            "streaming_first_token": "unavailable",
            "breakdown": breakdown,
        },
        "modes": {},
    }
    app = None
    runner = model_runner
    log_context = contextlib.nullcontext()
    needs_model = any(
        current_mode in {"offline", "auto"} or (current_mode == "online" and web_status.available)
        for current_mode in selected_modes
    )
    if runner is None and needs_model:
        log_context = contextlib.redirect_stderr(io.StringIO())
        with log_context:
            app = build_application()
            app.start()
        settings = app.settings
        runner = app.runtime_service.infer_model
    try:
        with log_context:
            for current_mode in selected_modes:
                if runner is None:
                    result["modes"][current_mode] = _run_real_mode(prompt, current_mode, lambda _request: _empty_model_response(), provider, web_status, settings)
                    continue
                result["modes"][current_mode] = _run_real_mode(prompt, current_mode, runner, provider, web_status, settings)
    finally:
        if app is not None:
            with log_context:
                app.stop()
    return result


def run_streaming_benchmark(
    *,
    root: str | Path | None = None,
    prompt: str = "hola jarvis",
    mode: str | None = None,
    stream_runner: Callable[[ModelRequest], Any] | None = None,
    web_provider: Any | None = None,
    debug_stream: bool = False,
) -> dict[str, Any]:
    selected_modes = [mode.strip().casefold()] if mode else ["offline", "auto", "online", "disabled"]
    project_root = Path(root).resolve(strict=False) if root else Path.cwd()
    provider = web_provider or build_web_search_provider()
    web_status = provider.status()
    result: dict[str, Any] = {
        "title": "Jarvis Streaming Benchmark",
        "prompt": prompt,
        "environment": {"root": str(project_root), "brave_key_configured": web_status.configured, "openai": "blocked", "gemini": "blocked", "debug_stream": debug_stream},
        "modes": {},
    }
    app = None
    runner = stream_runner
    log_context = contextlib.nullcontext()
    needs_model = any(current_mode in {"offline", "auto"} or (current_mode == "online" and web_status.available) for current_mode in selected_modes)
    if runner is None and needs_model:
        log_context = contextlib.redirect_stderr(io.StringIO())
        with log_context:
            app = build_application()
            app.start()
        runner = lambda request: app.runtime_service.stream_model(request)
    try:
        with log_context:
            settings = app.settings if app is not None else Settings()
            for current_mode in selected_modes:
                if current_mode == "disabled":
                    result["modes"][current_mode] = {"status": "ok", "provider": "none", "model": "none", "first_token_ms": None, "total_ms": 0.0, "chunks": 0, "response_chars": 46}
                elif runner is None:
                    result["modes"][current_mode] = {"status": "skipped", "reason": "streaming runner unavailable"}
                elif current_mode == "online":
                    result["modes"][current_mode] = _run_streaming_web_mode(prompt, current_mode, runner, provider, web_status, settings, debug_stream=debug_stream)
                elif current_mode == "auto" and should_use_web_search(prompt, mode="auto") and web_status.available:
                    result["modes"][current_mode] = _run_streaming_web_mode(prompt, current_mode, runner, provider, web_status, settings, debug_stream=debug_stream)
                else:
                    result["modes"][current_mode] = _run_streaming_model_mode(prompt, current_mode, runner, settings, web_used=False, debug_stream=debug_stream)
    finally:
        if app is not None:
            with log_context:
                app.stop()
    return result


def format_streaming_benchmark(result: dict[str, Any]) -> str:
    lines = [
        "Jarvis Streaming Benchmark",
        "",
        f"Prompt: {_safe_benchmark_prompt(str(result.get('prompt') or ''))}",
        f"Brave key: configured {str(result.get('environment', {}).get('brave_key_configured')).lower()}",
        "OpenAI: blocked",
        "Gemini: blocked",
        "",
    ]
    for mode, item in (result.get("modes") or {}).items():
        lines.append(f"{mode}:")
        for key in ("status", "provider", "model", "streaming_supported", "web_search_ms", "sources", "first_token_ms", "total_ms", "chunks", "response_chars", "reason", "suggestion"):
            if key in item:
                lines.append(f"  {key}: {item.get(key)}")
        debug = item.get("stream_debug")
        if debug:
            lines.append("  Stream debug:")
            for key in ("endpoint", "request_sent", "http_status", "first_line_ms", "first_content_ms", "lines_seen", "content_chunks", "done_seen", "empty_chunks", "parse_errors", "cancelled"):
                if key in debug:
                    lines.append(f"    {key}: {debug.get(key)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_real_benchmark(result: dict[str, Any]) -> str:
    show_breakdown = bool(result.get("environment", {}).get("breakdown"))
    lines = [
        "Jarvis Real Benchmark",
        "",
        f"Prompt: {_safe_benchmark_prompt(str(result.get('prompt') or ''))}",
        f"Brave key: configured {str(result.get('environment', {}).get('brave_key_configured')).lower()}",
        "OpenAI: blocked",
        "Gemini: blocked",
        "",
    ]
    for mode, item in (result.get("modes") or {}).items():
        lines.append(f"{mode}:")
        if item.get("status") == "skipped":
            lines.append("  status: skipped")
            lines.append(f"  reason: {item.get('reason')}")
            lines.append("")
            continue
        lines.append(f"  status: {item.get('status')}")
        lines.append(f"  provider: {item.get('provider')}")
        lines.append(f"  model: {item.get('model')}")
        lines.append(f"  total: {item.get('total_seconds')}s")
        lines.append(f"  response length: {item.get('response_length')} chars")
        lines.append(f"  web used: {'yes' if item.get('web_used') else 'no'}")
        if item.get("web_status"):
            lines.append(f"  web status: {item.get('web_status')}")
        if item.get("sources") is not None:
            lines.append(f"  sources: {item.get('sources')}")
        if item.get("synthesis_status"):
            lines.append(f"  synthesis: {item.get('synthesis_status')}")
        if item.get("reason"):
            lines.append(f"  reason: {item.get('reason')}")
        if item.get("message"):
            lines.append(f"  message: {item.get('message')}")
        if item.get("suggestion"):
            lines.append(f"  suggestion: {item.get('suggestion')}")
        breakdown = item.get("breakdown") or {}
        if show_breakdown and breakdown:
            lines.append("  breakdown:")
            for key in (
                "route_ms",
                "context_ms",
                "web_search_ms",
                "prompt_chars",
                "estimated_tokens",
                "generation_ms",
                "response_chars",
                "total_ms",
            ):
                if key in breakdown:
                    lines.append(f"    {key}: {breakdown.get(key)}")
        lines.append(f"  identity check: {item.get('identity_check')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _run_streaming_web_mode(prompt: str, mode: str, stream_runner: Callable[[ModelRequest], Any], web_provider: Any, web_status: Any, settings: Settings, *, debug_stream: bool = False) -> dict[str, Any]:
    if not web_status.available:
        return {"status": "skipped", "reason": "Brave Search API key not configured" if not web_status.configured else web_status.message}
    started = perf_counter()
    search_started = perf_counter()
    web_response = web_provider.search(prompt, max_results=web_provider.status().max_results)
    web_ms = round((perf_counter() - search_started) * 1000, 2)
    if web_response.status != "ok" or not web_response.hits:
        return {"status": "skipped", "reason": web_response.message or web_response.status, "web_search_ms": web_ms, "sources": len(web_response.hits)}
    prompt_with_sources = build_grounded_web_prompt(prompt, web_response, max_sources=settings.web_synthesis_max_sources, snippet_chars=settings.web_synthesis_snippet_chars)
    item = _run_streaming_model_mode(prompt_with_sources, mode, stream_runner, settings, web_used=True, started=started, debug_stream=debug_stream)
    item["web_search_ms"] = web_ms
    item["sources"] = len(web_response.hits)
    return item


def _safe_benchmark_prompt(prompt: str) -> str:
    sanitized = sanitize_web_query(prompt, max_chars=160)
    if not sanitized.allowed:
        return "[redacted]"
    return sanitized.query


def _run_streaming_model_mode(
    prompt: str,
    mode: str,
    stream_runner: Callable[[ModelRequest], Any],
    settings: Settings,
    *,
    web_used: bool,
    started: float | None = None,
    debug_stream: bool = False,
) -> dict[str, Any]:
    started_at = started if started is not None else perf_counter()
    profile = select_chat_generation_profile(prompt, settings, web_used=web_used, is_coding=False)
    request = ModelRequest(
        prompt=prompt,
        messages=[
            {"role": "system", "content": jarvis_identity_prompt(f"Benchmark streaming en modo {mode}. Responde breve.")},
            {"role": "user", "content": prompt},
        ],
        logical_model="general_assistant",
        task_type="assistant",
        required_capabilities=("chat",),
        temperature=profile.temperature,
        timeout_seconds=streaming_timeout_seconds(profile, settings),
        max_tokens=profile.max_tokens,
        stream=True,
        metadata={"source": "benchmark_stream", "mode": mode},
    )
    chunks = 0
    content = ""
    first_token_ms = None
    provider = "local"
    model = "gpt-oss:20b"
    streaming_supported = True
    stream_debug: dict[str, Any] | None = None
    done_reason: str | None = None
    try:
        for chunk in stream_runner(request):
            chunk = StreamChunk.model_validate(chunk)
            if chunk.error:
                reason = classify_local_model_error(Exception(chunk.error))
                if chunks == 0 and reason in {"ollama_timeout", "timeout"}:
                    reason = "first_token_timeout"
                return {"status": "error", "reason": reason, "provider": provider, "model": model, "chunks": chunks, "response_chars": len(content), "total_ms": round((perf_counter() - started_at) * 1000, 2)}
            provider = str(chunk.metadata.get("provider") or provider)
            model = str(chunk.metadata.get("model") or model)
            streaming_supported = bool(chunk.metadata.get("streaming_supported", streaming_supported))
            if chunk.metadata.get("stream_debug"):
                stream_debug = dict(chunk.metadata["stream_debug"])
            if chunk.metadata.get("reason"):
                done_reason = str(chunk.metadata["reason"])
            if chunk.text:
                if first_token_ms is None:
                    first_token_ms = round((perf_counter() - started_at) * 1000, 2)
                chunks += 1
                content += chunk.text
            if chunk.done:
                break
    except Exception as exc:  # noqa: BLE001
        reason = classify_local_model_error(exc)
        if chunks == 0 and reason in {"ollama_timeout", "timeout"}:
            reason = "first_token_timeout"
        return {"status": "error", "reason": reason, "provider": provider, "model": model, "chunks": chunks, "response_chars": len(content), "total_ms": round((perf_counter() - started_at) * 1000, 2)}
    item = {
        "status": "ok",
        "provider": provider,
        "model": model,
        "streaming_supported": streaming_supported,
        "first_token_ms": first_token_ms,
        "total_ms": round((perf_counter() - started_at) * 1000, 2),
        "chunks": chunks,
        "response_chars": len(sanitize_assistant_identity(content)),
    }
    if chunks == 0:
        reason = done_reason or "no_output"
        if stream_debug and stream_debug.get("parse_errors"):
            reason = "stream_parse_error"
        item.update(
            {
                "status": "error",
                "reason": reason,
                "suggestion": "retry with native Ollama stream or check model latency",
            }
        )
    if debug_stream and stream_debug:
        item["stream_debug"] = stream_debug
    return item


def _measure(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = perf_counter()
    value = fn()
    return value, round((perf_counter() - started) * 1000, 2)


def _run_real_mode(prompt: str, mode: str, model_runner: Callable[[ModelRequest], ModelResponse], web_provider: Any, web_status: Any, settings: Settings) -> dict[str, Any]:
    if mode == "disabled":
        started = perf_counter()
        content = "LLM disabled. Fallback determinista de Jarvis."
        total_ms = round((perf_counter() - started) * 1000, 2)
        return {
            "status": "ok",
            "provider": "none",
            "model": "none",
            "total_seconds": round(total_ms / 1000, 3),
            "response_length": len(content),
            "identity_check": _identity_check(content),
            "web_used": False,
            "first_token_seconds": None,
            "breakdown": {"route_ms": 0.0, "context_ms": 0.0, "generation_ms": 0.0, "response_chars": len(content), "total_ms": total_ms},
        }
    if mode == "offline":
        return _run_model_generation(prompt, mode, model_runner, web_used=False, settings=settings, route_ms=0.0, context_ms=0.0)
    if mode == "auto":
        route_started = perf_counter()
        use_web = should_use_web_search(prompt, mode="auto") and web_status.available
        route_ms = round((perf_counter() - route_started) * 1000, 2)
        if use_web:
            return _run_web_then_model(prompt, mode, model_runner, web_provider, settings, route_ms=route_ms)
        return _run_model_generation(prompt, mode, model_runner, web_used=False, settings=settings, route_ms=route_ms, context_ms=0.0)
    if mode == "online":
        if not web_status.available:
            return {
                "status": "skipped",
                "reason": "Brave Search API key not configured" if not web_status.configured else web_status.message,
                "provider": "ollama",
                "model": "gpt-oss:20b",
                "web_used": False,
                "identity_check": "not_run",
                "breakdown": {"route_ms": 0.0, "context_ms": 0.0, "generation_ms": 0.0, "total_ms": 0.0},
            }
        return _run_web_then_model(prompt, mode, model_runner, web_provider, settings, route_ms=0.0)
    return {"status": "skipped", "reason": f"invalid mode: {mode}", "identity_check": "not_run"}


def _run_web_then_model(prompt: str, mode: str, model_runner: Callable[[ModelRequest], ModelResponse], web_provider: Any, settings: Settings, *, route_ms: float) -> dict[str, Any]:
    started = perf_counter()
    search_started = perf_counter()
    web_response = web_provider.search(prompt, max_results=web_provider.status().max_results)
    search_seconds = round(perf_counter() - search_started, 3)
    web_search_ms = round(search_seconds * 1000, 2)
    if web_response.status != "ok" or not web_response.hits:
        return {
            "status": "skipped",
            "reason": web_response.message or web_response.status,
            "provider": "ollama",
            "model": "gpt-oss:20b",
            "web_status": web_response.status,
            "web_used": web_response.status == "ok",
            "sources": len(web_response.hits),
            "brave_seconds": search_seconds,
            "identity_check": "not_run",
            "breakdown": {"route_ms": route_ms, "web_search_ms": web_search_ms, "context_ms": 0.0, "generation_ms": 0.0, "total_ms": round((perf_counter() - started) * 1000, 2)},
        }
    context_started = perf_counter()
    prompt_with_sources = build_grounded_web_prompt(
        prompt,
        web_response,
        max_sources=settings.web_synthesis_max_sources,
        snippet_chars=settings.web_synthesis_snippet_chars,
    )
    context_ms = round((perf_counter() - context_started) * 1000, 2)
    item = _run_model_generation(prompt_with_sources, mode, model_runner, web_used=True, started=started, settings=settings, route_ms=route_ms, context_ms=context_ms)
    item["brave_seconds"] = search_seconds
    item["web_status"] = web_response.status
    item["sources"] = len(web_response.hits)
    item["synthesis_status"] = item.get("status")
    if item.get("status") == "error":
        item["message"] = local_model_failure_message(str(item.get("reason") or "ollama_error"), web_sources=len(web_response.hits))
    return item


def _run_model_generation(
    prompt: str,
    mode: str,
    model_runner: Callable[[ModelRequest], ModelResponse],
    *,
    web_used: bool,
    settings: Settings,
    route_ms: float,
    context_ms: float,
    started: float | None = None,
) -> dict[str, Any]:
    started_at = started if started is not None else perf_counter()
    profile = select_chat_generation_profile(prompt, settings, web_used=web_used, is_coding=False)
    prompt_chars = len(prompt or "")
    request = ModelRequest(
        prompt=prompt,
        messages=[
            {"role": "system", "content": jarvis_identity_prompt(f"Benchmark real en modo {mode}. Responde breve.")},
            {"role": "user", "content": prompt},
        ],
        logical_model="general_assistant",
        task_type="assistant",
        required_capabilities=("chat",),
        temperature=profile.temperature,
        timeout_seconds=profile.timeout_seconds,
        max_tokens=profile.max_tokens,
        metadata={"source": "benchmark_real", "mode": mode, "generation_profile": profile.name, "context_profile": profile.context_profile},
    )
    generation_started = perf_counter()
    try:
        response = model_runner(request)
    except Exception as exc:  # noqa: BLE001 - benchmark should report provider failures, not crash.
        reason = _safe_benchmark_error(exc)
        generation_ms = round((perf_counter() - generation_started) * 1000, 2)
        total_ms = round((perf_counter() - started_at) * 1000, 2)
        return {
            "status": "error",
            "reason": reason,
            "provider": "local",
            "model": "gpt-oss:20b",
            "total_seconds": round(perf_counter() - started_at, 3),
            "response_length": 0,
            "identity_check": "not_run",
            "web_used": web_used,
            "first_token_seconds": None,
            "message": local_model_failure_message(reason, web_sources=0),
            "suggestion": 'ollama run gpt-oss:20b "hola"',
            "breakdown": {
                "route_ms": route_ms,
                "context_ms": context_ms,
                "prompt_chars": prompt_chars,
                "estimated_tokens": estimated_tokens_from_chars(prompt_chars),
                "generation_ms": generation_ms,
                "response_chars": 0,
                "total_ms": total_ms,
            },
        }
    content = response.content or ""
    generation_ms = round((perf_counter() - generation_started) * 1000, 2)
    total_ms = round((perf_counter() - started_at) * 1000, 2)
    return {
        "status": "ok",
        "provider": response.provider_name,
        "model": response.model_name,
        "total_seconds": round(total_ms / 1000, 3),
        "provider_latency_ms": round(response.latency_ms, 2),
        "response_length": len(content),
        "identity_check": _identity_check(content),
        "web_used": web_used,
        "first_token_seconds": None,
        "breakdown": {
            "route_ms": route_ms,
            "context_ms": context_ms,
            "prompt_chars": prompt_chars,
            "estimated_tokens": estimated_tokens_from_chars(prompt_chars),
            "generation_ms": generation_ms,
            "response_chars": len(content),
            "total_ms": total_ms,
        },
    }


def _identity_check(content: str) -> str:
    lowered = content.casefold()
    if "chatgpt" in lowered:
        return "failed_chatgpt"
    if "soy gemini" in lowered or "soy openai" in lowered or "i am chatgpt" in lowered:
        return "failed_external_identity"
    return "ok"


def _safe_benchmark_error(exc: Exception) -> str:
    reason = classify_local_model_error(exc)
    if reason != "ollama_error":
        return reason
    text = str(exc)
    lowered = text.casefold()
    if any(token in lowered for token in ("token", "password", "secret", "credential", "api key", "apikey", ".env")):
        return "ollama_error"
    return "ollama_error"


def _empty_model_response() -> ModelResponse:
    return ModelResponse(
        provider_name="none",
        provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant",
        model_name="none",
        content="",
        latency_ms=0.0,
    )
