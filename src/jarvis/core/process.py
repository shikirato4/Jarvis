from __future__ import annotations

import subprocess
from typing import Protocol

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class ProcessRequest(JarvisBaseModel):
    command: list[str] = Field(min_length=1)
    cwd: str
    timeout_seconds: int = 30


class ProcessResult(JarvisBaseModel):
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class ProcessRunner(Protocol):
    def run(self, request: ProcessRequest) -> ProcessResult: ...


class SyncProcessRunner:
    def run(self, request: ProcessRequest) -> ProcessResult:
        completed = subprocess.run(
            request.command,
            cwd=request.cwd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=request.timeout_seconds,
            check=False,
        )
        return ProcessResult(
            command=request.command,
            cwd=request.cwd,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
