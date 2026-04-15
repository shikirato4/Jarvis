from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from jarvis.config import Settings
from jarvis.core.errors import SafetyViolationError
from jarvis.core.safety import ensure_within_roots

from .models import IndexSource


def resolve_source_root(source: IndexSource, settings: Settings) -> Path:
    if source.root_path is None:
        return settings.resolved_workspace_root
    allowed_roots = (
        settings.resolved_workspace_root,
        *settings.resolved_research_roots,
        *settings.resolved_system_search_roots,
    )
    return ensure_within_roots(source.root_path, allowed_roots, "indexing source")


def validate_source(source: IndexSource, settings: Settings) -> None:
    if source.root_path is not None:
        resolve_source_root(source, settings)


def is_sensitive_path(path: Path, settings: Settings, source: IndexSource) -> bool:
    lowered_path = str(path).casefold()
    lowered_name = path.name.casefold()
    for pattern in settings.indexing_sensitive_name_patterns:
        pattern = pattern.casefold()
        if fnmatch(lowered_name, pattern) or fnmatch(lowered_path, pattern):
            return True
    return bool(source.metadata.get("sensitive"))


def is_allowed_extension(path: Path, settings: Settings, source: IndexSource) -> bool:
    allowed = tuple(item.casefold() for item in (source.allowed_extensions or settings.indexing_allowed_extensions))
    if not allowed:
        return True
    return path.suffix.casefold() in allowed


def ensure_safe_scan(source: IndexSource, settings: Settings) -> Path | None:
    if source.root_path is None:
        return None
    root = resolve_source_root(source, settings)
    if root == Path(root.anchor):
        raise SafetyViolationError("refusing to index a filesystem root directly", details={"root_path": str(root)})
    return root
