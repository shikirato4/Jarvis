from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from jarvis.code_agent_runtime.base import AuthorizationContext, CommandRunResult
from jarvis.code_agent_runtime.paths import normalize_project_path, relative_to_root
from jarvis.code_agent_runtime.tools.terminal_runner import TerminalRunner


class GitManager:
    def __init__(self, project_root: Path, runner: TerminalRunner, *, max_diff_chars: int = 20_000) -> None:
        self._root = project_root.resolve(strict=False)
        self._runner = runner
        self._max_diff_chars = max_diff_chars

    def is_repo(self) -> bool:
        return (self._root / ".git").exists()

    def ensure_repo(self) -> None:
        if not self.is_repo():
            raise RuntimeError(f"not a git repository: {self._root}")

    def current_branch(self, authorization: AuthorizationContext) -> str:
        self.ensure_repo()
        result = self._run("git branch --show-current", authorization)
        return result.stdout.strip() or "HEAD"

    def status(self, authorization: AuthorizationContext) -> dict:
        self.ensure_repo()
        result = self._run("git status --short --branch", authorization)
        return {"branch": self._parse_branch(result.stdout), "raw": result.stdout, "return_code": result.return_code}

    def diff(self, authorization: AuthorizationContext, *, path: str | None = None) -> dict:
        self.ensure_repo()
        command = "git diff"
        argv = ["git", "diff"]
        if path:
            target = normalize_project_path(self._root, path)
            rel_path = relative_to_root(self._root, target)
            command = f"git diff -- {rel_path}"
            argv = ["git", "diff", "--", rel_path]
        result = self._run(command, authorization, argv=argv)
        text = result.stdout
        truncated = len(text) >= self._max_diff_chars
        return {"diff": text[: self._max_diff_chars], "truncated": truncated, "return_code": result.return_code}

    def diff_stat(self, authorization: AuthorizationContext) -> dict:
        self.ensure_repo()
        result = self._run("git diff --stat", authorization)
        return {"stat": result.stdout, "return_code": result.return_code}

    def changed_files(self, authorization: AuthorizationContext) -> list[str]:
        self.ensure_repo()
        result = self._run("git status --short", authorization)
        files: list[str] = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            files.append(line[3:].strip())
        return files

    def has_uncommitted_changes(self, authorization: AuthorizationContext) -> bool:
        return bool(self.changed_files(authorization))

    def branch_exists(self, name: str, authorization: AuthorizationContext) -> bool:
        self.ensure_repo()
        branch = self.sanitize_branch_name(name)
        result = self._run(f"git branch --list {branch}", authorization, argv=["git", "branch", "--list", branch])
        return bool(result.stdout.strip())

    def create_checkpoint(self, authorization: AuthorizationContext, *, message: str | None = None) -> dict:
        self.ensure_repo()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        checkpoint_message = self.sanitize_checkpoint_message(message or f"jarvis/checkpoint-{stamp}")
        command = f"git stash push -u -m {checkpoint_message}"
        result = self._run(command, authorization, argv=["git", "stash", "push", "-u", "-m", checkpoint_message])
        return {"kind": "stash", "message": checkpoint_message, "stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code}

    def create_branch(self, name: str, authorization: AuthorizationContext) -> dict:
        self.ensure_repo()
        branch = self.sanitize_branch_name(name)
        result = self._run(f"git switch -c {branch}", authorization, argv=["git", "switch", "-c", branch])
        return {"branch": branch, "stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code}

    def revert_file(self, path: str, authorization: AuthorizationContext) -> dict:
        self.ensure_repo()
        target = normalize_project_path(self._root, path)
        rel_path = relative_to_root(self._root, target)
        command = f"git checkout -- {rel_path}"
        result = self._run(command, authorization, argv=["git", "checkout", "--", rel_path])
        return {"path": rel_path, "stdout": result.stdout, "stderr": result.stderr, "return_code": result.return_code}

    def summary(self, authorization: AuthorizationContext) -> dict:
        self.ensure_repo()
        return {
            "branch": self.current_branch(authorization),
            "changed_files": self.changed_files(authorization),
            "diff_stat": self.diff_stat(authorization)["stat"],
        }

    @staticmethod
    def sanitize_checkpoint_message(message: str) -> str:
        folded = re.sub(r"[\r\n]+", " ", message).strip()
        folded = re.sub(r"(&&|\|\||[;|<>`$()\"'\\])", " ", folded)
        folded = re.sub(r"\brm\s+-[A-Za-z]*r[A-Za-z]*\b", " ", folded, flags=re.IGNORECASE)
        folded = re.sub(r"\b(del|rd|rmdir)\s+/(s|q)\b", " ", folded, flags=re.IGNORECASE)
        folded = re.sub(r"\b(format|shutdown|sudo|curl|wget)\b", " ", folded, flags=re.IGNORECASE)
        folded = re.sub(r"\s+", " ", folded).strip()
        return folded[:160] or "jarvis checkpoint"

    @staticmethod
    def sanitize_branch_name(name: str) -> str:
        raw = name.strip()
        if any(part in raw for part in ("..", "~", "^", ":", "\\", "@{")) or raw.startswith(("-", "/", ".")):
            raise ValueError(f"unsafe branch name: {name}")
        folded = raw.lower()
        folded = re.sub(r"[^a-z0-9._/-]+", "-", folded)
        folded = re.sub(r"-+", "-", folded).strip("-/.")
        folded = re.sub(r"/+", "/", folded)
        if not folded:
            raise ValueError("branch name cannot be empty")
        if not folded.startswith("jarvis/"):
            folded = f"jarvis/{folded}"
        if any(part in folded for part in ("..", "~", "^", ":", "\\", " ")):
            raise ValueError(f"unsafe branch name: {name}")
        parts = folded.split("/")
        if any(not part or part.startswith("-") or part.startswith(".") or part.endswith(".") for part in parts):
            raise ValueError(f"unsafe branch name: {name}")
        if folded.endswith(".lock") or folded.endswith("/") or folded.startswith("/") or folded == "jarvis":
            raise ValueError(f"unsafe branch name: {name}")
        return folded[:120]

    def _run(self, command: str, authorization: AuthorizationContext, *, argv: list[str] | None = None) -> CommandRunResult:
        argv = argv or self._argv(command)
        return self._runner.run(command, argv=argv, authorization=authorization)

    @staticmethod
    def _argv(command: str) -> list[str]:
        import shlex

        return shlex.split(command, posix=False)

    @staticmethod
    def _parse_branch(status: str) -> str:
        first = status.splitlines()[0] if status.splitlines() else ""
        if first.startswith("## "):
            return first[3:].split("...")[0].strip()
        return ""

    @staticmethod
    def _sanitize_message(message: str) -> str:
        return GitManager.sanitize_checkpoint_message(message)

    @staticmethod
    def _quote(value: str) -> str:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
