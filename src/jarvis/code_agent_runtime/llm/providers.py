from __future__ import annotations

import json
import os
from urllib import request as urlrequest

from jarvis.code_agent_runtime.llm.base import LLMProvider
from jarvis.code_agent_runtime.llm.config import LLMConfig
from jarvis.code_agent_runtime.llm.models import LLMGenerateRequest, LLMGenerateResult


class DisabledLLMProvider(LLMProvider):
    provider_name = "unavailable"
    model_name = "none"

    def is_available(self) -> bool:
        return False

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        return LLMGenerateResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            available=False,
            status="unavailable",
            error="Nemotron provider is unavailable",
        )


class FakeLLMProvider(LLMProvider):
    provider_name = "fake"

    def __init__(
        self,
        response: str | None = None,
        *,
        model_name: str = "fake-change-model",
        provider_name: str = "fake",
        env_var: str = "JARVIS_LLM_FAKE_RESPONSE",
    ) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self._response = response
        self._env_var = env_var

    def is_available(self) -> bool:
        return bool(
            self._response
            or os.getenv(self._env_var)
            or os.getenv("JARVIS_LLM_FAKE_RESPONSE")
        )

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        content = (
            self._response
            or os.getenv(self._env_var)
            or os.getenv("JARVIS_LLM_FAKE_RESPONSE", "")
        )
        if not content:
            return LLMGenerateResult(
                provider_name=self.provider_name,
                model_name=self.model_name,
                available=False,
                status="unavailable",
                error="fake provider response is not configured",
            )
        return LLMGenerateResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            available=True,
            status="ok",
            content=content[: request.max_output_chars],
        )


class NemotronCompatibleProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        api_key_env: str | None,
    ) -> None:
        self.provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._timeout = timeout_seconds
        self._api_key_env = api_key_env

    def is_available(self) -> bool:
        if not self._base_url or not self.model_name:
            return False
        return self._api_key_env is None or bool(os.getenv(self._api_key_env, ""))

    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        if not self.is_available():
            return LLMGenerateResult(
                provider_name=self.provider_name,
                model_name=self.model_name,
                available=False,
                status="unavailable",
                error=f"{self.provider_name} is not configured",
            )
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON for a reviewable patch proposal.",
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key_env is not None:
            headers["Authorization"] = f"Bearer {os.getenv(self._api_key_env, '')}"
        http_request = urlrequest.Request(
            self._base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=self._timeout) as response:  # noqa: S310
                body = json.loads(response.read().decode("utf-8", errors="replace"))
            content = str(body["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            return LLMGenerateResult(
                provider_name=self.provider_name,
                model_name=self.model_name,
                available=True,
                status="failed",
                error=type(exc).__name__,
            )
        return LLMGenerateResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            available=True,
            status="ok",
            content=content[: request.max_output_chars],
        )


def build_llm_provider(config: LLMConfig | None = None) -> LLMProvider:
    selected = config or LLMConfig.from_env_with_autodetect()
    return build_named_provider(selected.provider, selected, role="local")


def build_named_provider(name: str, config: LLMConfig, *, role: str) -> LLMProvider:
    selected = name.strip().casefold()
    if selected.startswith("fake"):
        return FakeLLMProvider(
            model_name=f"{role}-fake-change-model",
            provider_name=f"fake-{role}",
            env_var=f"JARVIS_{role.upper()}_LLM_FAKE_RESPONSE",
        )
    if selected == "nemotron_omni_local":
        return NemotronCompatibleProvider(
            provider_name=selected,
            base_url=config.local_base_url,
            model_name=config.local_model,
            timeout_seconds=config.timeout_seconds,
            api_key_env=None,
        )
    if selected == "nvidia_nim":
        return NemotronCompatibleProvider(
            provider_name=selected,
            base_url=config.online_base_url,
            model_name=config.online_model,
            timeout_seconds=config.timeout_seconds,
            api_key_env="NVIDIA_API_KEY",
        )
    return DisabledLLMProvider()
