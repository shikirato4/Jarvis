from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .errors import SafetyViolationError


def resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def ensure_within_roots(path: str | Path, roots: Iterable[str | Path], purpose: str) -> Path:
    resolved_path = resolve_path(path)
    resolved_roots = [resolve_path(root) for root in roots]
    if not resolved_roots:
        raise SafetyViolationError(f"no roots configured for {purpose}")
    for root in resolved_roots:
        if resolved_path == root or resolved_path.is_relative_to(root):
            return resolved_path
    roots_str = ", ".join(str(root) for root in resolved_roots)
    raise SafetyViolationError(f"path '{resolved_path}' is outside allowed roots for {purpose}: {roots_str}")


def ensure_allowed_executable(executable: str, allowlist: Iterable[str]) -> None:
    allowed = {item.casefold() for item in allowlist}
    if executable.casefold() not in allowed:
        raise SafetyViolationError(f"executable '{executable}' is not in the configured allowlist")
