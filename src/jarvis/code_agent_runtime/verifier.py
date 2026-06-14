from __future__ import annotations

from pathlib import Path


class CodeAgentVerifier:
    def file_exists(self, path: Path) -> bool:
        return path.exists() and path.is_file()

    def command_succeeded(self, return_code: int) -> bool:
        return return_code == 0
