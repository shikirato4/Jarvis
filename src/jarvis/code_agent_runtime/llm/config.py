from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from jarvis.models.base import JarvisBaseModel


LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
REMOTE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
_BLOCKED_HOSTS = {"api.openai.com", "generativelanguage.googleapis.com"}


class LLMConfig(JarvisBaseModel):
    provider: str = "provider_coordinator"
    model: str = REMOTE_MODEL
    base_url: str = "https://integrate.api.nvidia.com/v1"
    has_api_key: bool = False
    mode: str = "auto"
    local_provider: str = "nemotron_omni_local"
    online_provider: str = "nvidia_nim"
    local_base_url: str = "http://127.0.0.1:8000/v1"
    local_model: str = LOCAL_MODEL
    online_base_url: str = "https://integrate.api.nvidia.com/v1"
    online_model: str = REMOTE_MODEL
    prefer_local: bool = False
    allow_online_for_code: bool = True
    require_confirm_online: bool = False
    timeout_seconds: float = 90.0
    warning: str = ""
    external_api_keys_ignored: bool = False

    @classmethod
    def from_env(cls) -> LLMConfig:
        mode = cls._normalize_mode(os.getenv("JARVIS_LLM_MODE", "auto"))
        local_provider = os.getenv(
            "JARVIS_LOCAL_PROVIDER", "nemotron_omni_local"
        ).strip().casefold() or "nemotron_omni_local"
        online_provider = os.getenv(
            "JARVIS_ONLINE_PROVIDER", "nvidia_nim"
        ).strip().casefold() or "nvidia_nim"
        local_base_url = os.getenv(
            "LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1"
        ).strip()
        online_base_url = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip()
        local_model = os.getenv("LOCAL_LLM_MODEL", LOCAL_MODEL).strip() or LOCAL_MODEL
        online_model = os.getenv("NVIDIA_MODEL", REMOTE_MODEL).strip() or REMOTE_MODEL
        key_configured = bool(os.getenv("NVIDIA_API_KEY", ""))
        external_api_keys_ignored = any(
            bool(os.getenv(name, ""))
            for name in (
                "JARVIS_LLM_API_KEY",
                "OPENAI_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
            )
        )
        warning = ""
        if not cls._is_allowed_url(local_base_url):
            warning = "Local model base URL is invalid or blocked."
            local_base_url = ""
        if not cls._is_allowed_url(online_base_url):
            warning = cls._append_warning(warning, "NVIDIA base URL is invalid or blocked.")
            online_base_url = ""
        if external_api_keys_ignored:
            warning = cls._append_warning(
                warning,
                "Unrelated external API keys were ignored; only NVIDIA_API_KEY is used.",
            )
        provider = {
            "local": local_provider,
            "nvidia_api": online_provider,
            "auto": "provider_coordinator",
        }[mode]
        model = local_model if mode == "local" else online_model
        base_url = local_base_url if mode == "local" else online_base_url
        return cls(
            provider=provider,
            model=model,
            base_url=cls._safe_url(base_url) if base_url else "",
            has_api_key=key_configured,
            mode=mode,
            local_provider=local_provider,
            online_provider=online_provider,
            local_base_url=cls._safe_url(local_base_url) if local_base_url else "",
            local_model=local_model,
            online_base_url=cls._safe_url(online_base_url) if online_base_url else "",
            online_model=online_model,
            prefer_local=False,
            allow_online_for_code=True,
            require_confirm_online=False,
            warning=warning,
            external_api_keys_ignored=external_api_keys_ignored,
        )

    @classmethod
    def from_env_with_autodetect(
        cls,
        project_root: str | Path | None = None,
    ) -> LLMConfig:
        from jarvis.persistent_config import load_persistent_config

        config = cls.from_env()
        if "JARVIS_LLM_MODE" not in os.environ:
            persistent = load_persistent_config(project_root)
            config.mode = cls._normalize_mode(persistent.preferred_mode)
            config.provider = {
                "local": config.local_provider,
                "nvidia_api": config.online_provider,
                "auto": "provider_coordinator",
            }[config.mode]
        return config

    def safe_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": self.has_api_key,
            "mode": self.mode,
            "local_provider": self.local_provider,
            "online_provider": self.online_provider,
            "local_model": self.local_model,
            "online_model": self.online_model,
            "warning": self.warning,
            "external_api_keys_ignored": self.external_api_keys_ignored,
        }

    @staticmethod
    def _normalize_mode(value: str) -> str:
        normalized = str(value).strip().casefold()
        legacy_modes = {"offline", "online", "remote", "disabled"}
        if normalized in legacy_modes:
            return "auto"
        return normalized if normalized in {"auto", "nvidia_api", "local"} else "auto"

    @staticmethod
    def _is_allowed_url(value: str) -> bool:
        parsed = urlparse(value)
        host = (parsed.hostname or "").casefold()
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.netloc)
            and host not in _BLOCKED_HOSTS
            and not host.endswith(".openai.com")
            and not host.endswith(".googleapis.com")
        )

    @staticmethod
    def _safe_url(value: str) -> str:
        parsed = urlparse(value)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunparse((parsed.scheme, host, parsed.path.rstrip("/"), "", "", ""))

    @staticmethod
    def _append_warning(existing: str, message: str) -> str:
        return message if not existing else f"{existing}; {message}"
