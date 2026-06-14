from __future__ import annotations

import os
from pathlib import Path

from jarvis.code_agent_runtime.base import ProjectFileSummary, ProjectScanResult
from jarvis.code_agent_runtime.paths import is_ignored_dir, is_sensitive_path, relative_to_root


class ProjectScanner:
    def __init__(self, project_root: Path, *, max_files: int = 300) -> None:
        self._root = project_root
        self._max_files = max_files

    def scan(self) -> ProjectScanResult:
        directories: list[str] = []
        files: list[ProjectFileSummary] = []
        ignored: list[str] = []
        extension_counts: dict[str, int] = {}
        total_seen = 0

        for current, dirnames, filenames in os.walk(self._root):
            current_path = Path(current)
            kept_dirs: list[str] = []
            for dirname in dirnames:
                child = current_path / dirname
                if is_ignored_dir(child):
                    ignored.append(relative_to_root(self._root, child))
                else:
                    kept_dirs.append(dirname)
                    directories.append(relative_to_root(self._root, child))
            dirnames[:] = kept_dirs

            for filename in filenames:
                path = current_path / filename
                if is_sensitive_path(path):
                    continue
                total_seen += 1
                suffix = path.suffix.lower() or "<none>"
                extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
                if len(files) < self._max_files:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0
                    files.append(ProjectFileSummary(path=relative_to_root(self._root, path), size_bytes=size, extension=suffix))

        summary = (
            f"Project at {self._root} has {len(directories)} visible directories, "
            f"{total_seen} non-sensitive files seen, and {len(ignored)} ignored heavy directories."
        )
        return ProjectScanResult(
            root=str(self._root),
            directories=directories[: self._max_files],
            files=files,
            ignored_directories=ignored,
            total_files_seen=total_seen,
            total_files_listed=len(files),
            extension_counts=dict(sorted(extension_counts.items())),
            summary=summary,
        )
