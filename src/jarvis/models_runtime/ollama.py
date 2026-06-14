from __future__ import annotations

import json
import time
from typing import Any

import httpx

from jarvis.config import Settings

from .base import ModelProvider, ModelRequest, ModelResponse, ModelUsage, ProviderHealth, ProviderKind, StreamChunk


class OllamaProvider(ModelProvider):
    provider_name = "ollama"
    provider_kind = ProviderKind.LOCAL

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self._base_url)

    def health_check(self) -> ProviderHealth:
        try:
            response = self._client.get(
                self._settings.ollama_tags_endpoint,
                timeout=self._settings.ollama_healthcheck_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return ProviderHealth(
                provider_name=self.provider_name,
                healthy=True,
                details={"models": len(payload.get("models", []))},
            )
        except Exception as exc:
            return ProviderHealth(
                provider_name=self.provider_name,
                healthy=False,
                details={"error": str(exc)},
            )

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        attempts = max(self._settings.ollama_max_retries + 1, 1)
        timeout = timeout_seconds or self._settings.ollama_timeout_seconds
        payload = {
            "model": model_name,
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else request.temperature},
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.messages:
            payload["messages"] = [message.model_dump(mode="json") for message in request.messages]
        elif request.prompt:
            payload["messages"] = [{"role": "user", "content": request.prompt}]
        else:
            payload["messages"] = []

        last_error: Exception | None = None
        started_at = time.perf_counter()
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.post(
                    self._settings.ollama_chat_endpoint,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                body = response.json()
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                message = body.get("message", {})
                prompt_eval = body.get("prompt_eval_count")
                eval_count = body.get("eval_count")
                usage = ModelUsage(
                    input_tokens=prompt_eval,
                    output_tokens=eval_count,
                    total_tokens=(prompt_eval or 0) + (eval_count or 0) if prompt_eval is not None or eval_count is not None else None,
                )
                return ModelResponse(
                    provider_name=self.provider_name,
                    provider_kind=self.provider_kind,
                    logical_model=request.logical_model or model_name,
                    model_name=body.get("model", model_name),
                    content=message.get("content", ""),
                    raw=body,
                    latency_ms=elapsed_ms,
                    usage=usage,
                    metadata={"attempt": attempt, "task_type": request.task_type},
                )
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(self._settings.ollama_retry_backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    def stream_infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None, cancel_check=None):
        timeout = timeout_seconds or self._settings.ollama_timeout_seconds
        payload = {
            "model": model_name,
            "stream": True,
            "options": {"temperature": temperature if temperature is not None else request.temperature},
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.messages:
            payload["messages"] = [message.model_dump(mode="json") for message in request.messages]
        elif request.prompt:
            payload["messages"] = [{"role": "user", "content": request.prompt}]
        else:
            payload["messages"] = []
        started_at = time.perf_counter()
        chunk_count = 0
        lines_seen = 0
        empty_chunks = 0
        parse_errors = 0
        first_line_ms: float | None = None
        first_content_ms: float | None = None
        http_status: int | None = None
        debug_base = {
            "endpoint": "native",
            "request_sent": True,
            "cancelled": False,
        }
        with self._client.stream("POST", self._settings.ollama_chat_endpoint, json=payload, timeout=timeout) as response:
            http_status = response.status_code
            response.raise_for_status()
            for line in response.iter_lines():
                if cancel_check is not None and cancel_check():
                    debug = {
                        **debug_base,
                        "http_status": http_status,
                        "first_line_ms": first_line_ms,
                        "first_content_ms": first_content_ms,
                        "lines_seen": lines_seen,
                        "content_chunks": chunk_count,
                        "done_seen": False,
                        "empty_chunks": empty_chunks,
                        "parse_errors": parse_errors,
                        "cancelled": True,
                    }
                    yield StreamChunk(done=True, metadata={"cancelled": True, "chunks": chunk_count, "stream_debug": debug})
                    return
                if not line:
                    continue
                lines_seen += 1
                if first_line_ms is None:
                    first_line_ms = round((time.perf_counter() - started_at) * 1000, 2)
                try:
                    payload_line = json.loads(line)
                except Exception:
                    parse_errors += 1
                    continue
                message = payload_line.get("message") or {}
                text = message.get("content") or payload_line.get("response") or ""
                if text:
                    if first_content_ms is None:
                        first_content_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    chunk_count += 1
                    yield StreamChunk(text=text, metadata={"provider": self.provider_name, "model": model_name, "endpoint": "native"})
                else:
                    empty_chunks += 1
                if payload_line.get("done"):
                    debug = {
                        **debug_base,
                        "http_status": http_status,
                        "first_line_ms": first_line_ms,
                        "first_content_ms": first_content_ms,
                        "lines_seen": lines_seen,
                        "content_chunks": chunk_count,
                        "done_seen": True,
                        "empty_chunks": empty_chunks,
                        "parse_errors": parse_errors,
                        "cancelled": False,
                    }
                    yield StreamChunk(done=True, metadata={"chunks": chunk_count, "latency_ms": round((time.perf_counter() - started_at) * 1000, 2), "reason": "no_output" if chunk_count == 0 else None, "stream_debug": debug})
                    return
            debug = {
                **debug_base,
                "http_status": http_status,
                "first_line_ms": first_line_ms,
                "first_content_ms": first_content_ms,
                "lines_seen": lines_seen,
                "content_chunks": chunk_count,
                "done_seen": False,
                "empty_chunks": empty_chunks,
                "parse_errors": parse_errors,
                "cancelled": False,
            }
            yield StreamChunk(done=True, metadata={"chunks": chunk_count, "latency_ms": round((time.perf_counter() - started_at) * 1000, 2), "reason": "no_output" if chunk_count == 0 else None, "stream_debug": debug})
