from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

from jarvis.ollama_diagnostics import diagnose_ollama

logger = logging.getLogger(__name__)

@dataclass
class OllamaStatus:
    available: bool = False
    models: list[str] = field(default_factory=list)
    error: str | None = None
    status: str = "unknown"
    reachable: bool = False
    base_url: str = "http://127.0.0.1:11434"
    openai_base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "gpt-oss:20b"
    model_found: bool | None = None
    response_time_ms: float | None = None
    suggestions: list[str] = field(default_factory=list)

@dataclass
class EnvironmentStatus:
    internet_available: bool = False
    ollama: OllamaStatus = field(default_factory=OllamaStatus)
    recommended_mode: str = "disabled"
    recommended_local_provider: str | None = None
    recommended_local_model: str | None = None
    warnings: list[str] = field(default_factory=list)

def detect_internet(timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection(("8.8.8.8", 53), timeout=timeout):
            pass
        return True
    except OSError:
        return False

def detect_ollama(base_url: str = "http://127.0.0.1:11434", timeout: float = 1.5, prefer_model: str = "gpt-oss:20b") -> OllamaStatus:
    safe_base_url = _safe_http_url(base_url)
    if not safe_base_url:
        return OllamaStatus(available=False, error="Invalid Ollama base URL.", status="invalid_base_url")
    diagnostic = diagnose_ollama(base_url=safe_base_url, model=prefer_model, timeout_seconds=timeout)
    return OllamaStatus(
        available=diagnostic.available,
        models=diagnostic.models,
        error=None if diagnostic.status == "ok" else diagnostic.message,
        status=diagnostic.status,
        reachable=diagnostic.reachable,
        base_url=diagnostic.base_url,
        openai_base_url=diagnostic.openai_base_url,
        model=diagnostic.model,
        model_found=diagnostic.model_found,
        response_time_ms=diagnostic.response_time_ms,
        suggestions=diagnostic.suggestions,
    )

def detect_environment(ollama_base_url: str = "http://127.0.0.1:11434", prefer_model: str = "gpt-oss:20b", has_online_provider: bool = False) -> EnvironmentStatus:
    internet = detect_internet()
    ollama = detect_ollama(base_url=ollama_base_url, prefer_model=prefer_model)
    
    warnings = []
    recommended_mode = "disabled"
    recommended_local_model = None
    recommended_local_provider = None
    
    if ollama.available:
        recommended_local_provider = "ollama"
        if prefer_model in ollama.models:
            recommended_local_model = prefer_model
        elif ollama.models:
            recommended_local_model = ollama.models[0]
            warnings.append(f"Model '{prefer_model}' not found in Ollama. Using '{recommended_local_model}'.")
        else:
            warnings.append("Ollama is available but has no models installed.")
            
        if recommended_local_model:
            recommended_mode = "auto" if internet else "offline"
        else:
            recommended_mode = "disabled" if not has_online_provider else ("online" if internet else "disabled")
    else:
        if ollama.error:
            warnings.append(f"Ollama is not responding ({ollama.error}). Local models unavailable.")
        else:
            warnings.append("Ollama is not available. Local models unavailable.")
            
        if has_online_provider:
            recommended_mode = "online" if internet else "disabled"
        else:
            recommended_mode = "disabled"

    if not internet:
        warnings.append("No internet connection detected.")
        if recommended_mode == "online":
            recommended_mode = "disabled"
            warnings.append("Online mode requires internet. Falling back to disabled.")
            
    if recommended_mode == "disabled":
        warnings.append("No suitable LLM providers found. Agent operations will be disabled.")

    return EnvironmentStatus(
        internet_available=internet,
        ollama=ollama,
        recommended_mode=recommended_mode,
        recommended_local_provider=recommended_local_provider,
        recommended_local_model=recommended_local_model,
        warnings=warnings
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
    if any(term in lowered for term in ("token", "password", "secret", "credential", "api_key", "apikey", "key=")):
        return "Connection error."
    return value[:300]
