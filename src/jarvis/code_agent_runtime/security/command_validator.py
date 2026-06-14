from __future__ import annotations

import re
import shlex
from pathlib import Path

from jarvis.code_agent_runtime.base import RiskAssessment, RiskLevel
from jarvis.code_agent_runtime.paths import is_inside_project


class CommandValidator:
    _blocked_patterns = (
        r"\brm\s+-[^\n\r]*r",
        r"\bdel\s+/(s|q)",
        r"\brd\s+/(s|q)",
        r"\brmdir\s+/(s|q)",
        r"\bRemove-Item\b[^\n\r]*(?:-Recurse|-Force)",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\brestart(?:-computer)?\b",
        r"\bStart-Process\b[^\n\r]*-Verb\s+RunAs",
        r"\bsudo\b",
        r"\bchmod\s+777\b",
        r"^git\s+push\b",
        r"^git\s+remote\b",
        r"^git\s+reset\s+--hard\b",
        r"^git\s+clean\s+-[^\r\n]*[fd]",
        r"^git\s+rebase\b",
        r"^git\s+branch\s+-d\b",
        r"^git\s+branch\s+-D\b",
        r"\b(curl|wget)\b[^\n\r]*\|\s*(sh|bash|powershell|pwsh|cmd)\b",
        r"\b(type|cat|more)\b[^\n\r]*(\.env|id_rsa|token|credential|password|secret)",
    )
    _sensitive_patterns = (
        r"^npm\s+(install|i)\b",
        r"^pnpm\s+install\b",
        r"^yarn\s+install\b",
        r"^pip\s+install\b",
        r"^git\s+(checkout|switch|merge|reset|clean)\b",
        r"^git\s+stash\b",
        r"^git\s+commit\b",
        r"^git\s+add\b",
        r"^git\s+apply\b",
    )
    _safe_patterns = (
        r"^git\s+status$",
        r"^git\s+diff\b",
        r"^git\s+log\b",
        r"^git\s+branch\s+--show-current$",
        r"^git\s+branch\s+--list\b",
        r"^git\s+status\s+--short(?:\s+--branch)?$",
        r"^git\s+diff(?:\s+--stat)?(?:\s+--\s+[^\r\n]+)?$",
        r"^npm\s+run\s+(dev|build|test)$",
        r"^npm\s+test$",
        r"^pnpm\s+run\s+(dev|build)$",
        r"^pnpm\s+test$",
        r"^yarn\s+(dev|build|test)$",
        r"^python(?:\.exe)?\s+main\.py$",
        r"^python(?:\.exe)?\s+-m\s+pytest\b",
        r"^(?:[A-Za-z]:\\[^\r\n]*\\)?python(?:\.exe)?\s+-m\s+pytest\b",
        r"^(?:[A-Za-z]:\\[^\r\n]*\\)?python(?:\.exe)?\s+--version$",
        r"^pytest\b",
        r"^pip\s+list$",
        r"^node\s+--version$",
        r"^npm\s+--version$",
        r"^python(?:\.exe)?\s+--version$",
    )

    def validate(self, command: str, project_root: Path, *, cwd: Path | None = None) -> RiskAssessment:
        root = project_root.resolve(strict=False)
        effective_cwd = (cwd or root).resolve(strict=False)
        if not is_inside_project(root, effective_cwd):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="command cwd is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])

        stripped = command.strip()
        if not stripped:
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="empty command is blocked", requires_confirmation=True, requires_pin=True, tags=["empty_command"])
        if re.search(r"(^|\s)cd\s+(\.\.|/|~|[A-Za-z]:\\)", stripped, flags=re.IGNORECASE):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="command attempts to change outside project context", requires_confirmation=True, requires_pin=True, tags=["cwd_escape"])
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in self._blocked_patterns):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="blocked dangerous command pattern", requires_confirmation=True, requires_pin=True, tags=["dangerous_command"])
        if self._references_outside_path(stripped, root):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="command references a path outside the project", requires_confirmation=True, requires_pin=True, tags=["outside_project_path"])
        if re.search(r"(\&\&|\|\||;|>|>>)", stripped):
            return RiskAssessment(level=RiskLevel.SENSITIVE, reason="chained or redirected commands require confirmation and PIN", requires_confirmation=True, requires_pin=True, tags=["compound_command"])
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in self._sensitive_patterns):
            return RiskAssessment(level=RiskLevel.SENSITIVE, reason="dependency, branch or cleanup command requires confirmation and PIN", requires_confirmation=True, requires_pin=True, tags=["sensitive_command"])
        if any(re.search(pattern, stripped, flags=re.IGNORECASE) for pattern in self._safe_patterns):
            return RiskAssessment(level=RiskLevel.MINOR_CHANGE, reason="command is in the project safe allowlist", tags=["safe_command"])
        return RiskAssessment(level=RiskLevel.SENSITIVE, reason="unknown command requires confirmation and PIN", requires_confirmation=True, requires_pin=True, tags=["unknown_command"])

    def to_argv(self, command: str) -> list[str]:
        return shlex.split(command, posix=False)

    def _references_outside_path(self, command: str, project_root: Path) -> bool:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return True
        for token in tokens[1:]:
            normalized = token.strip("\"'")
            if normalized in {"", ".", "--"} or normalized.startswith("-"):
                continue
            if normalized == ".." or normalized.startswith("../") or normalized.startswith("..\\") or "/../" in normalized or "\\..\\" in normalized:
                return True
            candidate = Path(normalized).expanduser()
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=False)
                if not is_inside_project(project_root, resolved):
                    return True
        return False
