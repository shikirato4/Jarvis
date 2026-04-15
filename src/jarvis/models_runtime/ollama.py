from __future__ import annotations

import time
from typing import Any

import httpx

from jarvis.config import Settings

from .base import ModelProvider, ModelRequest, ModelResponse, ModelUsage, ProviderHealth, ProviderKind


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
