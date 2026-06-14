from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from urllib.parse import urlparse, urlunparse

import httpx


@dataclass
class OllamaDiagnostic:
    reachable: bool
    available: bool
    base_url: str
    openai_base_url: str
    model: str
    model_found: bool | None
    response_time_ms: float | None
    status: str
    message: str
    models: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def diagnose_ollama(
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "gpt-oss:20b",
    timeout_seconds: float = 2.0,
    client: httpx.Client | None = None,
) -> OllamaDiagnostic:
    safe_base_url = _safe_http_url(base_url) or "http://127.0.0.1:11434"
    started = perf_counter()
    suggestions = _suggestions(model)
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http_client.get(f"{safe_base_url.rstrip('/')}/api/tags", timeout=timeout_seconds)
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        response.raise_for_status()
        payload = response.json()
        models = _extract_model_names(payload)
        found = model in models
        status = "ok" if found else "model_missing"
        message = "ok" if found else f"Ollama esta disponible, pero no encontre el modelo {model}."
        return OllamaDiagnostic(
            reachable=True,
            available=found,
            base_url=safe_base_url,
            openai_base_url=f"{safe_base_url.rstrip('/')}/v1",
            model=model,
            model_found=found,
            response_time_ms=elapsed_ms,
            status=status,
            message=message,
            models=models,
            suggestions=suggestions if not found else [],
        )
    except httpx.TimeoutException:
        return _failed(
            status="timeout",
            message="Ollama no respondio antes del timeout.",
            base_url=safe_base_url,
            model=model,
            response_time_ms=round((perf_counter() - started) * 1000, 2),
            suggestions=suggestions,
        )
    except httpx.ConnectError:
        return _failed(
            status="connection_refused",
            message="No pude conectar con Ollama en la URL configurada.",
            base_url=safe_base_url,
            model=model,
            response_time_ms=round((perf_counter() - started) * 1000, 2),
            suggestions=suggestions,
        )
    except Exception as exc:  # noqa: BLE001
        return _failed(
            status=_classify_ollama_error(exc),
            message=_safe_error_message(str(exc)),
            base_url=safe_base_url,
            model=model,
            response_time_ms=round((perf_counter() - started) * 1000, 2),
            suggestions=suggestions,
        )
    finally:
        if owns_client:
            http_client.close()


def classify_local_model_error(exc: Exception) -> str:
    return _classify_ollama_error(exc)


def local_model_failure_message(reason: str, *, web_sources: int = 0) -> str:
    if web_sources > 0:
        prefix = f"Encontre {web_sources} fuente(s) con Brave Search, pero el modelo local de Jarvis no respondio para redactar la respuesta."
    else:
        prefix = "El modelo local de Jarvis no respondio."
    if reason == "ollama_timeout":
        detail = "Ollama tardo demasiado en responder."
    elif reason == "ollama_connection_refused":
        detail = "No pude conectar con Ollama en http://127.0.0.1:11434."
    elif reason == "ollama_model_missing":
        detail = "No encontre el modelo configurado gpt-oss:20b en Ollama."
    else:
        detail = "La solicitud local fallo antes de generar respuesta."
    return f"{prefix}\n\n{detail}\n\nSugerencia: ejecuta `ollama run gpt-oss:20b \"hola\"` y vuelve a intentar."


def _failed(
    *,
    status: str,
    message: str,
    base_url: str,
    model: str,
    response_time_ms: float | None,
    suggestions: list[str],
) -> OllamaDiagnostic:
    return OllamaDiagnostic(
        reachable=False,
        available=False,
        base_url=base_url,
        openai_base_url=f"{base_url.rstrip('/')}/v1",
        model=model,
        model_found=None,
        response_time_ms=response_time_ms,
        status=status,
        message=message,
        suggestions=suggestions,
    )


def _extract_model_names(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            names.append(name)
    return names


def _classify_ollama_error(exc: Exception) -> str:
    lowered = f"{type(exc).__name__}: {exc}".casefold()
    if any(token in lowered for token in ("timeout", "timed out", "readtimeout", "connecttimeout")):
        return "ollama_timeout"
    if any(token in lowered for token in ("connection refused", "winerror 10061", "connecterror", "connection failed")):
        return "ollama_connection_refused"
    if any(token in lowered for token in ("model", "not found", "404")):
        return "ollama_model_missing"
    return "ollama_error"


def _safe_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunparse((parsed.scheme, host, parsed.path.rstrip("/"), "", "", ""))


def _safe_error_message(value: str) -> str:
    lowered = value.casefold()
    if any(term in lowered for term in ("token", "password", "secret", "credential", "api_key", "apikey", "key=", ".env")):
        return "Ollama request failed."
    return value[:300]


def _suggestions(model: str) -> list[str]:
    return [
        "ollama list",
        "ollama ps",
        f'ollama run {model} "hola"',
    ]


def warmup_ollama(
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "gpt-oss:20b",
    prompt: str = "ping",
    timeout_seconds: float = 20.0,
    client: httpx.Client | None = None,
) -> OllamaDiagnostic:
    safe_base_url = _safe_http_url(base_url) or "http://127.0.0.1:11434"
    started = perf_counter()
    owns_client = client is None
    http_client = client or httpx.Client(timeout=timeout_seconds)
    try:
        response = http_client.post(
            f"{safe_base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt[:80] or "ping"}],
                "options": {"num_predict": 8, "temperature": 0},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        return OllamaDiagnostic(
            reachable=True,
            available=True,
            base_url=safe_base_url,
            openai_base_url=f"{safe_base_url.rstrip('/')}/v1",
            model=model,
            model_found=True,
            response_time_ms=elapsed_ms,
            status="ok",
            message="warmup ok",
        )
    except Exception as exc:  # noqa: BLE001
        return _failed(
            status=_classify_ollama_error(exc).removeprefix("ollama_") or "error",
            message=_safe_error_message(str(exc)),
            base_url=safe_base_url,
            model=model,
            response_time_ms=round((perf_counter() - started) * 1000, 2),
            suggestions=_suggestions(model),
        )
    finally:
        if owns_client:
            http_client.close()
