from __future__ import annotations

from pathlib import Path

from jarvis.code_agent_runtime.base import FileReadResult
from jarvis.code_agent_runtime.paths import is_sensitive_path, looks_like_text_path, normalize_project_path, relative_to_root


class FileReader:
    def __init__(self, project_root: Path, *, max_bytes: int = 120_000) -> None:
        self._root = project_root
        self._max_bytes = max_bytes

    def read(self, path: str) -> FileReadResult:
        resolved = normalize_project_path(self._root, path)
        if is_sensitive_path(resolved):
            raise PermissionError(f"sensitive file is blocked: {relative_to_root(self._root, resolved)}")
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        size = resolved.stat().st_size
        if self._is_binary(resolved):
            raise ValueError(f"binary file is blocked: {relative_to_root(self._root, resolved)}")
        with resolved.open("rb") as handle:
            raw = handle.read(self._max_bytes + 1)
        truncated = len(raw) > self._max_bytes
        content = raw[: self._max_bytes].decode("utf-8", errors="replace")
        return FileReadResult(path=relative_to_root(self._root, resolved), size_bytes=size, truncated=truncated, content=content)

    def _is_binary(self, path: Path) -> bool:
        if looks_like_text_path(path):
            return False
        try:
            sample = path.read_bytes()[:4096]
        except OSError:
            return True
        return b"\x00" in sample
