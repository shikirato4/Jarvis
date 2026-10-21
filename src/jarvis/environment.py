from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import httpx

from jarvis.models_runtime.model_identity import ConfiguredModelIdentity


@dataclass
class LocalEndpointStatus:
    available: bool = False
    reachable: bool = False
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-model"
    model_found: bool | None = None
    response_time_ms: float | None = None
    status: str = "unavailable"
    error: str | None = None
    models: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class EnvironmentStatus:
    internet_available: bool = False
    local_endpoint: LocalEndpointStatus = field(default_factory=LocalEndpointStatus)
    nvidia_configured: bool = False
    recommended_mode: str = "auto"
    recommended_local_provider: str = "qwen_local"
    recommended_local_model: str = (
        "local-model"
    )
    warnings: list[str] = field(default_factory=list)


def detect_internet(timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            return True
    except OSError:
        return False


def detect_local_endpoint(
    base_url: str = "http://127.0.0.1:8000/v1",
    *,
    model: str = "local-model",
    model_aliases: tuple[str, ...] = (),
    timeout: float = 1.5,
    client: httpx.Client | None = None,
) -> LocalEndpointStatus:
    safe_base_url = _safe_http_url(base_url)
    if not safe_base_url:
        return LocalEndpointStatus(
            base_url="",
            model=model,
            status="invalid_base_url",
            error="Invalid local OpenAI-compatible base URL.",
        )
    started_at = time.perf_counter()
    owns_client = client is None
    http_client = client or httpx.Client(base_url=safe_base_url)
    try:
        response = http_client.get("/models", timeout=timeout)
        response.raise_for_status()
        body = response.json()
        models = [
            str(item.get("id"))
            for item in body.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
        identity = ConfiguredModelIdentity(model, model_aliases)
        found = identity.select_advertised(models) is not None
        suggestions = [] if found else [
            f"Configura el endpoint para anunciar {identity.canonical_model} o un alias permitido."
        ]
        return LocalEndpointStatus(
            available=found,
            reachable=True,
            base_url=safe_base_url,
            model=model,
            model_found=found,
            response_time_ms=round((time.perf_counter() - started_at) * 1000, 2),
            status="ok" if found else "model_missing",
            error=None if found else "The configured local model is not exposed by the endpoint.",
            models=models,
            suggestions=suggestions,
        )
    except httpx.TimeoutException:
        return _local_failure(safe_base_url, model, started_at, "timeout", "Local endpoint timed out.")
    except (httpx.HTTPError, ValueError) as exc:
        return _local_failure(
            safe_base_url,
            model,
            started_at,
            "connection_failed",
            _safe_error_message(str(exc)),
        )
    finally:
        if owns_client:
            http_client.close()


def detect_environment(
    local_base_url: str = "http://127.0.0.1:8000/v1",
    local_model: str = "local-model",
    local_model_aliases: tuple[str, ...] = (),
    *,
    nvidia_configured: bool = False,
) -> EnvironmentStatus:
    internet = detect_internet()
    local = detect_local_endpoint(
        local_base_url,
        model=local_model,
        model_aliases=local_model_aliases,
    )
    warnings: list[str] = []
    if not local.available:
        warnings.append(local.error or "Local model endpoint unavailable.")
    if not nvidia_configured:
        warnings.append("NVIDIA API key is not configured.")
    if not internet:
        warnings.append("No internet connection detected; NVIDIA API may be unavailable.")
    if not local.available and not nvidia_configured:
        warnings.append("Both configured model routes are currently unavailable; automatic mode has no usable route.")
    return EnvironmentStatus(
        internet_available=internet,
        local_endpoint=local,
        nvidia_configured=nvidia_configured,
        warnings=warnings,
    )


def _local_failure(
    base_url: str,
    model: str,
    started_at: float,
    status: str,
    error: str,
) -> LocalEndpointStatus:
    return LocalEndpointStatus(
        available=False,
        reachable=False,
        base_url=base_url,
        model=model,
        response_time_ms=round((time.perf_counter() - started_at) * 1000, 2),
        status=status,
        error=error,
        suggestions=["Inicia o corrige el endpoint local OpenAI-compatible configurado."],
    )


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
    if any(
        term in lowered
        for term in ("token", "password", "secret", "credential", "api_key", "apikey", "key=")
    ):
        return "Connection error."
    return value[:300]
