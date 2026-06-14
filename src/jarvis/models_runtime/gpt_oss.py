from __future__ import annotations

import json
import time
from typing import Any

import httpx

from jarvis.config import Settings

from .base import ModelProvider, ModelRequest, ModelResponse, ModelUsage, ProviderHealth, ProviderKind, StreamChunk


class GptOssProvider(ModelProvider):
    provider_name = "gpt_oss"
    provider_kind = ProviderKind.REMOTE

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        self._base_url = settings.gpt_oss_base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if settings.gpt_oss_api_key:
            headers["Authorization"] = f"Bearer {settings.gpt_oss_api_key}"
        self._client = client or httpx.Client(base_url=self._base_url, headers=headers)

    def health_check(self) -> ProviderHealth:
        try:
            response = self._client.get(
                self._endpoint(self._settings.gpt_oss_models_endpoint),
                timeout=self._settings.gpt_oss_healthcheck_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") or payload.get("models") or []
            return ProviderHealth(
                provider_name=self.provider_name,
                healthy=True,
                details={"models": len(models)},
            )
        except Exception as exc:
            return ProviderHealth(
                provider_name=self.provider_name,
                healthy=False,
                details={"error": str(exc)},
            )

    def infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None) -> ModelResponse:
        timeout = timeout_seconds or self._settings.gpt_oss_timeout_seconds
        payload: dict[str, Any] = {
            "model": model_name,
            "stream": False,
            "temperature": temperature if temperature is not None else request.temperature,
            "messages": self._messages_payload(request),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        started_at = time.perf_counter()
        response = self._client.post(
            self._endpoint(self._settings.gpt_oss_chat_endpoint),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        choice = ((body.get("choices") or [{}])[0] or {})
        message = choice.get("message") or {}
        usage_payload = body.get("usage") or {}
        usage = ModelUsage(
            input_tokens=usage_payload.get("prompt_tokens"),
            output_tokens=usage_payload.get("completion_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
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
            metadata={"task_type": request.task_type},
        )

    def stream_infer(self, request: ModelRequest, *, model_name: str, temperature: float | None, timeout_seconds: float | None, cancel_check=None):
        timeout = timeout_seconds or self._settings.gpt_oss_timeout_seconds
        payload: dict[str, Any] = {
            "model": model_name,
            "stream": True,
            "temperature": temperature if temperature is not None else request.temperature,
            "messages": self._messages_payload(request),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        started_at = time.perf_counter()
        chunk_count = 0
        lines_seen = 0
        empty_chunks = 0
        parse_errors = 0
        first_line_ms: float | None = None
        first_content_ms: float | None = None
        http_status: int | None = None
        debug_base = {
            "endpoint": "openai_compatible",
            "request_sent": True,
            "cancelled": False,
        }
        with self._client.stream(
            "POST",
            self._endpoint(self._settings.gpt_oss_chat_endpoint),
            json=payload,
            timeout=timeout,
        ) as response:
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
                data = line[5:].strip() if line.startswith("data:") else line.strip()
                if not data:
                    continue
                if data == "[DONE]":
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
                try:
                    payload_line = json.loads(data)
                except Exception:
                    parse_errors += 1
                    continue
                choice = ((payload_line.get("choices") or [{}])[0] or {})
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    if first_content_ms is None:
                        first_content_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    chunk_count += 1
                    yield StreamChunk(text=text, metadata={"provider": self.provider_name, "model": model_name, "endpoint": "openai_compatible"})
                else:
                    empty_chunks += 1
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

    @staticmethod
    def _messages_payload(request: ModelRequest) -> list[dict[str, str]]:
        if request.messages:
            return [message.model_dump(mode="json") for message in request.messages]
        if request.prompt:
            return [{"role": "user", "content": request.prompt}]
        return []

    @staticmethod
    def _endpoint(path: str) -> str:
        return path.lstrip("/")
