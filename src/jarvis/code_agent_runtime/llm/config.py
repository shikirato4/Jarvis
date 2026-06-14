from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from jarvis.models.base import JarvisBaseModel


_IGNORED_OFFICIAL_KEY_ENV_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
_BLOCKED_OFFICIAL_PROVIDERS = {"openai", "gemini", "google", "google-gemini"}
_BLOCKED_OFFICIAL_HOSTS = {
    "api.openai.com",
    "generativelanguage.googleapis.com",
}


class LLMConfig(JarvisBaseModel):
    provider: str = "disabled"
    model: str = ""
    base_url: str = ""
    has_api_key: bool = False
    mode: str = "disabled"
    local_provider: str = "ollama"
    online_provider: str = "openai-compatible"
    prefer_local: bool = True
    allow_online_for_code: bool = False
    require_confirm_online: bool = True
    timeout_seconds: float = 30.0
    warning: str = ""
    external_api_keys_ignored: bool = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("JARVIS_LLM_PROVIDER", "disabled").strip().casefold() or "disabled"
        model = os.getenv("JARVIS_LLM_MODEL", "").strip()
        base_url = os.getenv("JARVIS_LLM_BASE_URL", "").strip()
        api_key = os.getenv("JARVIS_LLM_API_KEY", "")
        mode = os.getenv("JARVIS_LLM_MODE", provider if provider in {"auto", "offline", "online", "disabled"} else ("auto" if provider != "disabled" else "disabled")).strip().casefold()
        local_provider = os.getenv("JARVIS_LOCAL_PROVIDER", "ollama").strip().casefold() or "ollama"
        online_provider = os.getenv("JARVIS_ONLINE_PROVIDER", "openai-compatible").strip().casefold() or "openai-compatible"
        if provider == "fake" and "JARVIS_LOCAL_PROVIDER" not in os.environ:
            local_provider = "fake"
        warning = ""
        external_api_keys_ignored = cls._has_ignored_official_api_keys()
        if external_api_keys_ignored:
            warning = cls._append_warning(warning, "OpenAI/Gemini/Google API keys are ignored by Jarvis policy")
        if cls.is_official_provider_blocked(provider):
            warning = cls._append_warning(warning, f"provider {provider} is blocked by Jarvis API key policy")
            provider = "disabled"
            mode = "disabled"
        if cls.is_official_provider_blocked(online_provider):
            warning = cls._append_warning(warning, f"online provider {online_provider} is blocked by Jarvis API key policy")
            online_provider = "disabled"
        if base_url and not cls._is_allowed_url(base_url):
            warning = cls._append_warning(warning, "LLM base URL is not allowed; official OpenAI/Gemini hosts and non-http endpoints are blocked")
            base_url = ""
        if mode not in {"auto", "offline", "online", "disabled"}:
            warning = cls._append_warning(warning, "invalid LLM mode, using disabled")
            mode = "disabled"
        return cls(
            provider=provider,
            model=model,
            base_url=cls._safe_url(base_url) if base_url else "",
            has_api_key=bool(api_key),
            mode=mode,
            local_provider=local_provider,
            online_provider=online_provider,
            prefer_local=cls._env_bool("JARVIS_LLM_PREFER_LOCAL", True),
            allow_online_for_code=cls._env_bool("JARVIS_ALLOW_ONLINE_FOR_CODE", False),
            require_confirm_online=cls._env_bool("JARVIS_REQUIRE_CONFIRM_ONLINE", True),
            warning=warning,
            external_api_keys_ignored=external_api_keys_ignored,
        )

    @classmethod
    def from_env_with_autodetect(cls, project_root: str | Path | None = None) -> "LLMConfig":
        from jarvis.environment import detect_environment
        from jarvis.persistent_config import load_persistent_config
        
        # If any of the main env vars are explicitly set, respect them and use traditional from_env
        if any(os.getenv(k) for k in ["JARVIS_LLM_PROVIDER", "JARVIS_LLM_MODEL", "JARVIS_LLM_BASE_URL", "JARVIS_LLM_MODE"]):
            return cls.from_env()
            
        p_config = load_persistent_config(project_root)
        api_key = os.getenv("JARVIS_LLM_API_KEY", "")
        
        env_status = detect_environment(
            ollama_base_url=p_config.local_base_url,
            prefer_model=p_config.local_model,
            has_online_provider=bool(api_key)
        )
        
        mode = p_config.preferred_mode
        if mode == "auto" and env_status.recommended_mode == "disabled":
            mode = "disabled"
            
        provider = env_status.recommended_local_provider or p_config.local_provider
        model = env_status.recommended_local_model or p_config.local_model
        warning = "; ".join(env_status.warnings)
        external_api_keys_ignored = cls._has_ignored_official_api_keys()
        if external_api_keys_ignored:
            warning = cls._append_warning(warning, "OpenAI/Gemini/Google API keys are ignored by Jarvis policy")
        base_url = p_config.local_base_url
        if base_url and not cls._is_allowed_url(base_url):
            warning = cls._append_warning(warning, "LLM base URL is not allowed; official OpenAI/Gemini hosts and non-http endpoints are blocked")
            base_url = ""
        
        if env_status.recommended_mode == "disabled" and mode != "disabled":
            warning = cls._append_warning(warning, f"Environment suggests disabled mode. Proceeding with {mode} but it may fail.")
            
        if not provider or not model:
            provider = "disabled"
            mode = "disabled"
            
        # In case the provider is disabled, keep mode consistent
        if mode == "disabled":
            provider = "disabled"
            model = ""
            
        return cls(
            provider=provider,
            model=model,
            base_url=cls._safe_url(base_url) if base_url else "",
            has_api_key=bool(api_key),
            mode=mode,
            local_provider=p_config.local_provider,
            online_provider="openai-compatible",
            prefer_local=True,
            allow_online_for_code=p_config.allow_online_for_public_tasks,
            require_confirm_online=p_config.require_confirm_online,
            warning=warning.strip("; "),
            external_api_keys_ignored=external_api_keys_ignored,
        )

    def safe_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": self.has_api_key,
            "mode": self.mode,
            "local_provider": self.local_provider,
            "online_provider": self.online_provider,
            "prefer_local": self.prefer_local,
            "allow_online_for_code": self.allow_online_for_code,
            "require_confirm_online": self.require_confirm_online,
            "warning": self.warning,
            "external_api_keys_ignored": self.external_api_keys_ignored,
        }

    @staticmethod
    def _is_allowed_url(value: str) -> bool:
        parsed = urlparse(value)
        host = (parsed.hostname or "").casefold()
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not LLMConfig._is_blocked_official_host(host)

    @staticmethod
    def is_official_provider_blocked(provider: str) -> bool:
        return provider.strip().casefold() in _BLOCKED_OFFICIAL_PROVIDERS

    @staticmethod
    def _is_blocked_official_host(host: str) -> bool:
        if not host:
            return False
        if host in _BLOCKED_OFFICIAL_HOSTS:
            return True
        return host.endswith(".openai.com") or host.endswith(".googleapis.com")

    @staticmethod
    def _safe_url(value: str) -> str:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunparse((parsed.scheme, host, parsed.path, "", "", ""))

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().casefold() in {"1", "true", "yes", "on"}

    @staticmethod
    def _has_ignored_official_api_keys() -> bool:
        return any(bool(os.getenv(name)) for name in _IGNORED_OFFICIAL_KEY_ENV_VARS)

    @staticmethod
    def _append_warning(existing: str, message: str) -> str:
        if not existing:
            return message
        if message in existing:
            return existing
        return f"{existing}; {message}"
