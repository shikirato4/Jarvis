from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
_SECRET_TERMS = ("key", "token", "password", "secret", "pin", "credential", "private")

@dataclass
class PersistentConfig:
    preferred_mode: str = "auto"
    local_provider: str = "ollama"
    local_base_url: str = "http://127.0.0.1:11434"
    local_model: str = "gpt-oss:20b"
    allow_online_for_public_tasks: bool = False
    require_confirm_online: bool = True
    
    @classmethod
    def defaults(cls) -> PersistentConfig:
        return cls()

def get_config_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root or Path.cwd()).expanduser().resolve(strict=False)
    config_dir = root / "runtime"
    return config_dir / "jarvis_config.json"

def load_persistent_config(project_root: str | Path | None = None) -> PersistentConfig:
    config_path = get_config_path(project_root)
    if not config_path.exists():
        defaults = PersistentConfig.defaults()
        save_persistent_config(defaults, project_root)
        return defaults
        
    try:
        content = config_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("persistent config must be a JSON object")
        return PersistentConfig(**_sanitize_payload(data))
    except Exception as e:
        backup_path = _backup_corrupt_config(config_path)
        logger.warning("Failed to parse persistent config, using defaults: %s; backup=%s", e, backup_path)
        defaults = PersistentConfig.defaults()
        save_persistent_config(defaults, project_root)
        return defaults

def save_persistent_config(config: PersistentConfig, project_root: str | Path | None = None) -> None:
    config_path = get_config_path(project_root)
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    safe_data = _sanitize_payload(asdict(config))
    
    try:
        config_path.write_text(json.dumps(safe_data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error("Failed to save persistent config: %s", e)


def _sanitize_payload(data: dict) -> dict:
    valid_keys = PersistentConfig.__dataclass_fields__.keys()
    safe_data = {}
    for key, value in data.items():
        normalized_key = str(key).casefold()
        if key not in valid_keys or _looks_sensitive(normalized_key) or _looks_sensitive(value):
            continue
        safe_data[key] = value
    return safe_data


def _looks_sensitive(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    return any(term in normalized for term in _SECRET_TERMS)


def _backup_corrupt_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = config_path.with_name(f"{config_path.name}.corrupt-{timestamp}.bak")
    try:
        shutil.copy2(config_path, backup_path)
        return backup_path
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to back up corrupt persistent config: %s", e)
        return None
