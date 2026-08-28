from __future__ import annotations

from pathlib import Path

from platformdirs import user_config_path, user_data_path, user_log_path


APP_NAME = "Jarvis"


def user_data_dir() -> Path:
    return Path(user_data_path(APP_NAME, appauthor=False, roaming=False))


def user_config_dir() -> Path:
    return Path(user_config_path(APP_NAME, appauthor=False, roaming=False))


def user_logs_dir() -> Path:
    return Path(user_log_path(APP_NAME, appauthor=False))


def resolve_user_data_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)
    return (user_data_dir() / expanded).resolve(strict=False)
