from __future__ import annotations

import fnmatch
from pathlib import Path

IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".idea",
    ".vscode",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".turbo",
    "__pycache__",
    "coverage",
    ".coverage",
    "runtime",
}

IGNORED_DIR_PATTERNS = (
    ".venv*",
    ".pytest_tmp*",
    "pytest-cache-files-*",
    "runtime_perf_*",
    "runtime_test_*",
    "runtime_ui_*",
)

SENSITIVE_EXACT_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SENSITIVE_NAME_PARTS = (
    "secret",
    "token",
    "credential",
    "credentials",
    "private",
    "passwd",
    "password",
    "apikey",
    "api_key",
    "access_key",
)

SENSITIVE_EXTENSIONS = {
    ".pem",
    ".key",
    ".crt",
    ".cer",
    ".pfx",
    ".p12",
    ".jks",
    ".keystore",
}

TEXT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cmd",
    ".css",
    ".csv",
    ".env.example",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    return Path(project_root or Path.cwd()).expanduser().resolve(strict=False)


def normalize_project_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve(strict=False)
    if not is_inside_project(project_root, resolved):
        raise PermissionError(f"path is outside project root: {resolved}")
    return resolved


def is_inside_project(project_root: Path, path: Path) -> bool:
    root = project_root.resolve(strict=False)
    resolved = path.resolve(strict=False)
    return resolved == root or resolved.is_relative_to(root)


def is_ignored_dir(path: Path) -> bool:
    name = path.name
    folded = name.casefold()
    if folded in {item.casefold() for item in IGNORED_DIR_NAMES}:
        return True
    return any(fnmatch.fnmatch(folded, pattern.casefold()) for pattern in IGNORED_DIR_PATTERNS)


def is_sensitive_path(path: Path) -> bool:
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    if name in {item.casefold() for item in SENSITIVE_EXACT_NAMES}:
        return True
    if suffix in {item.casefold() for item in SENSITIVE_EXTENSIONS}:
        return True
    return any(part in name for part in SENSITIVE_NAME_PARTS)


def looks_like_text_path(path: Path) -> bool:
    name = path.name.casefold()
    suffix = path.suffix.casefold()
    return suffix in TEXT_EXTENSIONS or name in TEXT_EXTENSIONS


def relative_to_root(project_root: Path, path: Path) -> str:
    return str(path.resolve(strict=False).relative_to(project_root.resolve(strict=False)))
