from __future__ import annotations

import json
import os
from urllib import request as urlrequest

from jarvis.code_agent_runtime.llm.base import LLMProvider
from jarvis.code_agent_runtime.llm.config import LLMConfig
from jarvis.code_agent_runtime.llm.models import LLMGenerateRequest, LLMGenerateResult


class DisabledLLMProvider(LLMProvider):
    provider_name = "disabled"
    model_name = "none"

    def is_available(self) -> bool:
        return False

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=False, status="unavailable", error="LLM provider is disabled")


class FakeLLMProvider(LLMProvider):
    provider_name = "fake"

    def __init__(self, response: str | None = None, *, model_name: str = "fake-change-model", provider_name: str = "fake", env_var: str = "JARVIS_LLM_FAKE_RESPONSE") -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._response = response
        self._env_var = env_var

    def is_available(self) -> bool:
        return bool(self._response or os.getenv(self._env_var) or os.getenv("JARVIS_LLM_FAKE_RESPONSE"))

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        content = self._response or os.getenv(self._env_var) or os.getenv("JARVIS_LLM_FAKE_RESPONSE", "")
        if not content:
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=False, status="unavailable", error="fake provider response is not configured")
        return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="ok", content=content[: request.max_output_chars])


class OllamaLLMProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, config: LLMConfig) -> None:
        self._base_url = (os.getenv("JARVIS_LOCAL_LLM_BASE_URL") or os.getenv("JARVIS_OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model_name = os.getenv("JARVIS_LOCAL_LLM_MODEL") or config.model or "gpt-oss:20b"
        self._timeout = config.timeout_seconds

    def is_available(self) -> bool:
        return bool(self._base_url and self.model_name and os.getenv("JARVIS_LOCAL_LLM_DISABLED", "").casefold() not in {"1", "true", "yes"})

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if not self.is_available():
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=False, status="unavailable", error="local Ollama provider is not configured")
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "Return only valid JSON for a reviewable patch proposal."},
                {"role": "user", "content": request.prompt},
            ],
            "stream": False,
        }
        http_request = urlrequest.Request(
            self._base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=self._timeout) as response:  # noqa: S310 - local URL comes from safe runtime config.
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="failed", error=str(exc))
        content = str(data.get("message", {}).get("content", ""))
        if not content:
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="failed", error="Ollama response did not include message content")
        return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="ok", content=content[: request.max_output_chars])


class OpenAICompatibleProvider(LLMProvider):
    provider_name = "openai-compatible"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self.model_name = config.model or "unknown"

    def is_available(self) -> bool:
        return bool(self._config.base_url and self._config.model and self._config.has_api_key and LLMConfig._is_allowed_url(self._config.base_url))

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if not self.is_available():
            return LLMGenerateResult(
                provider_name=self.provider_name,
                model_name=self.model_name,
                available=False,
                status="unavailable",
                error="OpenAI-compatible provider requires JARVIS_LLM_BASE_URL, JARVIS_LLM_MODEL and JARVIS_LLM_API_KEY",
            )
        api_key = os.getenv("JARVIS_LLM_API_KEY", "")
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON for a reviewable patch proposal."},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0,
        }
        endpoint = self._config.base_url.rstrip("/") + "/chat/completions"
        http_request = urlrequest.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=self._config.timeout_seconds) as response:  # noqa: S310 - URL is user-configured and validated.
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="failed", error=str(exc))
        content = ""
        try:
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="failed", error="provider response did not include message content")
        return LLMGenerateResult(provider_name=self.provider_name, model_name=self.model_name, available=True, status="ok", content=content[: request.max_output_chars])


def build_llm_provider(config: LLMConfig | None = None) -> LLMProvider:
    selected = config or LLMConfig.from_env_with_autodetect()
    if selected.provider == "fake":
        return FakeLLMProvider(model_name=selected.model or "fake-change-model")
    if selected.provider in {"ollama", "local"}:
        return OllamaLLMProvider(selected)
    if LLMConfig.is_official_provider_blocked(selected.provider):
        return DisabledLLMProvider()
    if selected.provider in {"openai-compatible", "lmstudio", "lm-studio"}:
        return OpenAICompatibleProvider(selected)
    return DisabledLLMProvider()


def build_named_provider(name: str, config: LLMConfig, *, role: str) -> LLMProvider:
    selected = name.strip().casefold()
    if selected in {"fake", "fake-local", "local-fake"}:
        return FakeLLMProvider(model_name=config.model or f"{role}-fake-change-model", provider_name=f"fake-{role}", env_var=f"JARVIS_{role.upper()}_LLM_FAKE_RESPONSE")
    if selected in {"fake-online", "online-fake"}:
        return FakeLLMProvider(model_name=config.model or "online-fake-change-model", provider_name="fake-online", env_var="JARVIS_ONLINE_LLM_FAKE_RESPONSE")
    if selected in {"ollama", "local"}:
        return OllamaLLMProvider(config)
    if LLMConfig.is_official_provider_blocked(selected):
        return DisabledLLMProvider()
    if selected in {"openai-compatible", "lmstudio", "lm-studio"}:
        return OpenAICompatibleProvider(config)
    return DisabledLLMProvider()
