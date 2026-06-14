from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from jarvis.code_agent_runtime.change_generator.models import ResolvedTarget
from jarvis.code_agent_runtime.llm.base import LLMProvider
from jarvis.code_agent_runtime.llm.config import LLMConfig
from jarvis.code_agent_runtime.llm.providers import build_named_provider


SECRET_MARKERS = (
    ".env",
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "apikey",
    "private key",
    "certificate",
    ".pem",
    ".key",
    "id_rsa",
)


@dataclass(frozen=True)
class SensitivityResult:
    level: str
    reason: str


@dataclass(frozen=True)
class RouteDecision:
    mode: str
    provider_kind: str
    provider_name: str
    model_name: str
    sensitivity: str
    reason: str
    allowed: bool
    fallback_used: bool = False
    warning: str = ""


class LLMRouter:
    def __init__(self, config: LLMConfig, *, local_provider: LLMProvider | None = None, online_provider: LLMProvider | None = None) -> None:
        self._config = config
        self._local_provider = local_provider or build_named_provider(config.local_provider, config, role="local")
        self._online_provider = online_provider or build_named_provider(config.online_provider, config, role="online")

    @property
    def local_provider(self) -> LLMProvider:
        return self._local_provider

    @property
    def online_provider(self) -> LLMProvider:
        return self._online_provider

    def classify(self, task: str, targets: Iterable[ResolvedTarget] = (), *, prompt: str = "") -> SensitivityResult:
        blob = " ".join([task, prompt, " ".join(target.path for target in targets)]).casefold()
        if any(marker in blob for marker in SECRET_MARKERS):
            return SensitivityResult("secret", "secret marker detected")
        if any(target.exists for target in targets):
            return SensitivityResult("sensitive", "project file context detected")
        if any(part in blob for part in ("c:/users/", "c:\\users\\", "/home/", "runtime/code_agent", "project_memory")):
            return SensitivityResult("sensitive", "local private path or memory context detected")
        if any(part in blob for part in ("src/", "tests/", ".py", ".ts", ".tsx", ".js")):
            return SensitivityResult("internal", "local coding task detected")
        return SensitivityResult("public", "no private context detected")

    def route(self, task: str, targets: list[ResolvedTarget], *, mode: str | None = None, allow_online_override: bool = False, prompt: str = "") -> tuple[LLMProvider | None, RouteDecision]:
        selected_mode = (mode or self._config.mode).strip().casefold()
        sensitivity = self.classify(task, targets, prompt=prompt)
        if selected_mode == "disabled":
            return None, self._decision(selected_mode, "none", None, sensitivity, "LLM disabled by mode", False)
        if sensitivity.level == "secret":
            return None, self._decision(selected_mode, "none", None, sensitivity, "secret context is never sent to an LLM", False, warning="deterministic fallback required")
        if selected_mode == "offline":
            if self._local_provider.is_available():
                return self._local_provider, self._decision(selected_mode, "local", self._local_provider, sensitivity, "offline mode uses only local provider", True)
            return None, self._decision(selected_mode, "local", self._local_provider, sensitivity, "offline local provider unavailable", False, warning="deterministic fallback required")
        if selected_mode == "online":
            if self._local_provider.is_available():
                return self._local_provider, self._decision(
                    selected_mode,
                    "local",
                    self._local_provider,
                    sensitivity,
                    "online mode uses Brave Search for web and local Ollama for generation; external online LLMs are blocked",
                    True,
                )
            return None, self._decision(selected_mode, "local", self._local_provider, sensitivity, "online mode requires local Ollama generation, but local provider is unavailable", False, warning="deterministic fallback required")
        if selected_mode != "auto":
            return None, self._decision("disabled", "none", None, sensitivity, "invalid LLM mode", False)
        if sensitivity.level in {"sensitive", "internal"} and self._config.prefer_local:
            if self._local_provider.is_available():
                return self._local_provider, self._decision("auto", "local", self._local_provider, sensitivity, "auto prefers local for project code context", True)
            if sensitivity.level == "sensitive" and not (self._config.allow_online_for_code or allow_online_override):
                return None, self._decision("auto", "local", self._local_provider, sensitivity, "local unavailable and online blocked for sensitive context", False, fallback_used=True)
        if self._local_provider.is_available():
            return self._local_provider, self._decision("auto", "local", self._local_provider, sensitivity, "auto selected local fallback", True, fallback_used=True)
        return None, self._decision("auto", "none", None, sensitivity, "no LLM provider available", False, fallback_used=True, warning="deterministic fallback required")

    @staticmethod
    def _decision(mode: str, kind: str, provider: LLMProvider | None, sensitivity: SensitivityResult, reason: str, allowed: bool, *, fallback_used: bool = False, warning: str = "") -> RouteDecision:
        return RouteDecision(
            mode=mode,
            provider_kind=kind,
            provider_name=provider.provider_name if provider else "none",
            model_name=provider.model_name if provider else "none",
            sensitivity=sensitivity.level,
            reason=reason,
            allowed=allowed,
            fallback_used=fallback_used,
            warning=warning,
        )
