from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from jarvis.runtime_paths import user_config_dir


logger = logging.getLogger(__name__)

LOCAL_PROVIDER = "qwen_local"
LOCAL_BASE_URL = "http://127.0.0.1:8081/v1"
LOCAL_MODEL = "Qwen3.5-9B-IQ4_XS"
REASONING_PREFERENCE = "auto"
_SECRET_TERMS = ("key", "token", "password", "secret", "pin", "credential", "private")
_LEGACY_MARKERS = ("oll" + "ama", "gpt" + "_oss", "gpt" + "-oss")
_PRE_RUNTIME_MANAGER_URL = "http://127.0.0.1:8000/v1"
_PRE_RUNTIME_MANAGER_MODEL = "Qwen 3.8 quantized"


class PersistentConfigWriteError(OSError):
    """The session setting changed, but durable storage could not be updated."""


@dataclass
class PersistentConfig:
    preferred_mode: str = "auto"
    local_provider: str = LOCAL_PROVIDER
    local_base_url: str = LOCAL_BASE_URL
    local_model: str = LOCAL_MODEL
    local_model_aliases: tuple[str, ...] = ()
    reasoning_preference: str = REASONING_PREFERENCE
    allow_online_for_public_tasks: bool = False
    require_confirm_online: bool = True
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def defaults(cls) -> PersistentConfig:
        return cls()


def get_config_path(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        return user_config_dir() / "jarvis_config.json"
    root = Path(project_root).expanduser().resolve(strict=False)
    return root / "runtime" / "jarvis_config.json"


def load_persistent_config(project_root: str | Path | None = None) -> PersistentConfig:
    """Load and normalize configuration without mutating the filesystem."""

    config_path = get_config_path(project_root)
    if not config_path.exists():
        return PersistentConfig.defaults()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("persistent config must be a JSON object")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "persistent_config_read_failed",
            extra={"config_path": str(config_path), "error_type": type(exc).__name__},
        )
        return PersistentConfig.defaults()
    return _config_from_payload(data)


def save_persistent_config(
    config: PersistentConfig,
    project_root: str | Path | None = None,
) -> None:
    """Atomically replace the durable configuration with a redacted payload."""

    config_path = get_config_path(project_root)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload_from_config(config)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary_path = config_path.with_name(
        f".{config_path.name}.{uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, mode="w", encoding="utf-8", newline="\n") as temporary:
            descriptor = None
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, config_path)
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PersistentConfigWriteError(
            f"No se pudo guardar la configuración en {config_path}."
        ) from exc


def _config_from_payload(data: dict[str, Any]) -> PersistentConfig:
    safe = _sanitize_payload(data)
    known_names = {
        "preferred_mode",
        "local_provider",
        "local_base_url",
        "local_model",
        "local_model_aliases",
        "reasoning_preference",
        "allow_online_for_public_tasks",
        "require_confirm_online",
    }
    known = {key: value for key, value in safe.items() if key in known_names}
    extra = {key: value for key, value in safe.items() if key not in known_names}
    migrated = _migrate_payload(known)
    aliases = migrated.get("local_model_aliases", ())
    if isinstance(aliases, list):
        migrated["local_model_aliases"] = tuple(str(item) for item in aliases)
    elif not isinstance(aliases, tuple):
        migrated["local_model_aliases"] = ()
    return PersistentConfig(**migrated, extra=extra)


def _payload_from_config(config: PersistentConfig) -> dict[str, Any]:
    payload = dict(config.extra)
    payload.update(
        {
            "preferred_mode": _normalize_mode(config.preferred_mode),
            "local_provider": LOCAL_PROVIDER,
            "local_base_url": (
                LOCAL_BASE_URL
                if _has_legacy_marker(config.local_base_url)
                or ":11434" in config.local_base_url
                else config.local_base_url
            ),
            "local_model": _safe_local_model(config.local_model),
            "local_model_aliases": list(config.local_model_aliases),
            "reasoning_preference": _normalize_reasoning(config.reasoning_preference),
            "allow_online_for_public_tasks": bool(config.allow_online_for_public_tasks),
            "require_confirm_online": bool(config.require_confirm_online),
        }
    )
    return _sanitize_payload(payload)


def _sanitize_payload(data: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key)
        if _looks_sensitive(key) or _looks_sensitive(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) or item is None for item in value
        ):
            safe[key] = value
        elif isinstance(value, dict):
            safe[key] = _sanitize_payload(value)
    return safe


def _migrate_payload(data: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(data)
    mode = str(migrated.get("preferred_mode", "auto")).strip().casefold()
    legacy_provider = _has_legacy_marker(str(migrated.get("local_provider", "")))
    legacy_model = _has_legacy_marker(str(migrated.get("local_model", "")))
    legacy_url = ":11434" in str(migrated.get("local_base_url", ""))
    pre_runtime_manager_url = (
        str(migrated.get("local_base_url", "")).rstrip("/").casefold()
        == _PRE_RUNTIME_MANAGER_URL.casefold()
    )
    pre_runtime_manager_model = (
        str(migrated.get("local_model", "")).strip().casefold()
        == _PRE_RUNTIME_MANAGER_MODEL.casefold()
    )
    legacy_fields: list[str] = []
    if legacy_provider:
        legacy_fields.append("local_provider")
    if legacy_model:
        legacy_fields.append("local_model")
    if legacy_url:
        legacy_fields.append("local_base_url")
    if mode in {"offline", "online", "remote", "disabled"}:
        legacy_fields.append("preferred_mode")
    if (
        mode in {"offline", "online", "remote", "disabled"}
        or legacy_provider
        or legacy_model
        or legacy_url
    ):
        mode = "auto"
    migrated["preferred_mode"] = _normalize_mode(mode)
    migrated["local_provider"] = LOCAL_PROVIDER
    migrated["local_base_url"] = (
        LOCAL_BASE_URL
        if legacy_url or pre_runtime_manager_url
        else str(migrated.get("local_base_url") or LOCAL_BASE_URL)
    )
    migrated["local_model"] = (
        LOCAL_MODEL
        if legacy_model or pre_runtime_manager_model
        else _safe_local_model(str(migrated.get("local_model") or LOCAL_MODEL))
    )
    if pre_runtime_manager_url:
        legacy_fields.append("local_base_url_pre_runtime_manager")
    if pre_runtime_manager_model:
        legacy_fields.append("local_model_pre_runtime_manager")
    aliases = migrated.get("local_model_aliases", ())
    if not isinstance(aliases, (list, tuple)):
        aliases = ()
    migrated["local_model_aliases"] = tuple(
        str(alias).strip() for alias in aliases if str(alias).strip()
    )
    migrated["reasoning_preference"] = _normalize_reasoning(migrated.get("reasoning_preference", REASONING_PREFERENCE))
    if legacy_fields:
        logger.warning(
            "persistent_config_legacy_detected",
            extra={"legacy_fields": tuple(sorted(set(legacy_fields)))},
        )
    return migrated


def _normalize_reasoning(value: object) -> str:
    normalized = str(value or REASONING_PREFERENCE).strip().casefold()
    return normalized if normalized in {"off", "low", "medium", "xhigh", "auto"} else REASONING_PREFERENCE


def _safe_local_model(value: str) -> str:
    from jarvis.models_runtime.model_identity import ConfiguredModelIdentity

    try:
        return ConfiguredModelIdentity(value).configured_model
    except ValueError:
        return LOCAL_MODEL


def _normalize_mode(value: str) -> str:
    normalized = str(value).strip().casefold()
    return normalized if normalized in {"auto", "nvidia_api", "local"} else "auto"


def _has_legacy_marker(value: str) -> bool:
    normalized = value.casefold()
    return any(marker in normalized for marker in _LEGACY_MARKERS)


def _looks_sensitive(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    return any(term in normalized for term in _SECRET_TERMS)
