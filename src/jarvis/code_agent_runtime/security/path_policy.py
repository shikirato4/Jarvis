from __future__ import annotations

import fnmatch
from pathlib import Path

from jarvis.code_agent_runtime.paths import is_inside_project, is_sensitive_path

PROTECTED_PROJECT_FILES = {
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "requirements.txt",
    "pyproject.toml",
    "tsconfig.json",
    "docker-compose.yml",
    "dockerfile",
}

PROTECTED_PROJECT_PATTERNS = (
    "vite.config.*",
    "next.config.*",
    "*.deploy.*",
    "*.deployment.*",
    "*security*",
    "*.ps1",
    "*.sh",
    "*.bat",
    "*.cmd",
)

SYSTEM_ROOTS = (
    Path("C:/Windows"),
    Path("C:/Program Files"),
    Path("C:/Program Files (x86)"),
)


class PathPolicy:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve(strict=False)

    def is_allowed_project_path(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return is_inside_project(self._root, resolved) and not self.is_system_path(resolved)

    def is_system_path(self, path: Path) -> bool:
        resolved = path.resolve(strict=False)
        return any(resolved == root.resolve(strict=False) or resolved.is_relative_to(root.resolve(strict=False)) for root in SYSTEM_ROOTS)

    def is_sensitive(self, path: Path) -> bool:
        return is_sensitive_path(path) or path.suffix.casefold() in {".db", ".sqlite", ".sqlite3"}

    def is_protected_project_file(self, path: Path) -> bool:
        name = path.name.casefold()
        if name in PROTECTED_PROJECT_FILES:
            return True
        return any(fnmatch.fnmatch(name, pattern.casefold()) for pattern in PROTECTED_PROJECT_PATTERNS)
