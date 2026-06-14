from __future__ import annotations

import subprocess
from pathlib import Path

from jarvis.code_agent_runtime.base import AuthorizationContext, CodeActionKind, CommandRunResult


class TerminalRunner:
    def __init__(self, project_root: Path, *, timeout_seconds: int = 120, max_output_chars: int = 20_000) -> None:
        self._root = project_root
        self._timeout_seconds = timeout_seconds
        self._max_output_chars = max_output_chars

    def run(
        self,
        command: str,
        *,
        argv: list[str] | None = None,
        dry_run: bool = False,
        authorization: AuthorizationContext | None = None,
    ) -> CommandRunResult:
        self._require_authorization(command, authorization)
        if dry_run:
            return CommandRunResult(command=command, cwd=str(self._root), return_code=0, stdout="", stderr="", timed_out=False)
        try:
            completed = subprocess.run(
                argv or [command],
                cwd=str(self._root),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
            return CommandRunResult(
                command=command,
                cwd=str(self._root),
                return_code=completed.returncode,
                stdout=completed.stdout[-self._max_output_chars :],
                stderr=completed.stderr[-self._max_output_chars :],
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return CommandRunResult(
                command=command,
                cwd=str(self._root),
                return_code=124,
                stdout=stdout[-self._max_output_chars :],
                stderr=stderr[-self._max_output_chars :],
                timed_out=True,
            )

    @staticmethod
    def _require_authorization(command: str, authorization: AuthorizationContext | None) -> None:
        if authorization is None:
            raise PermissionError("command execution requires an authorization context")
        if not authorization.allowed:
            raise PermissionError("command execution authorization is not allowed")
        if authorization.action not in {CodeActionKind.COMMAND_RUN, CodeActionKind.GIT_OPERATION}:
            raise PermissionError("authorization action does not allow command execution")
        if authorization.target != command:
            raise PermissionError("authorization target does not match command")
        if authorization.risk.requires_confirmation and not authorization.confirmation_confirmed:
            raise PermissionError("command execution requires confirmed authorization")
        if authorization.risk.requires_pin and authorization.pin_verified is not True:
            raise PermissionError("command execution requires verified PIN authorization")
