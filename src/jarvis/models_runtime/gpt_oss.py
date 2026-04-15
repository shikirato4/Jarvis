from __future__ import annotations

import time
from typing import Any

import httpx

from jarvis.config import Settings

from .base import ModelProvider, ModelRequest, ModelResponse, ModelUsage, ProviderHealth, ProviderKind


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
