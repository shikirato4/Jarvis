from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

from .models import SelfImprovementIssue

_GUIDED_FIX_RE = re.compile(
    r'jarvis-self-improve:\s*replace\s+"(?P<old>.+?)"\s*=>\s*"(?P<new>.+?)"',
    re.IGNORECASE,
)


class SelfImprovementAnalyzer:
    def __init__(self, *, ops_runtime=None, research_runtime=None, log_file: Path | None = None, logger: logging.Logger | None = None) -> None:
        self._ops_runtime = ops_runtime
        self._research_runtime = research_runtime
        self._log_file = log_file
        self._logger = logger or logging.getLogger("jarvis.self_improvement.analyzer")

    def analyze_code(self, path: Path) -> dict[str, object]:
        target = path.resolve()
        files = self._collect_python_files(target)
        failure_history = self._collect_failure_history()
        issues = self.detect_issues(files, failure_history=failure_history)
        return {
            "path": str(target),
            "files": [str(item) for item in files],
            "issues": issues,
            "failure_history": failure_history,
            "research_context": self._research_context(),
        }

    def detect_issues(self, files: list[Path], *, failure_history: list[str]) -> list[SelfImprovementIssue]:
        issues: list[SelfImprovementIssue] = []
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            issues.extend(self._guided_fix_issues(file_path, text))
            issues.extend(self._static_issues(file_path, text, failure_history=failure_history))
        issues.sort(key=lambda item: (0 if item.auto_fixable else 1, item.file_path, item.line or 0))
        return issues

    def _guided_fix_issues(self, file_path: Path, text: str) -> list[SelfImprovementIssue]:
        issues: list[SelfImprovementIssue] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = _GUIDED_FIX_RE.search(line)
            if not match:
                continue
            issues.append(
                SelfImprovementIssue(
                    issue_id=f"guided:{file_path}:{line_number}",
                    file_path=str(file_path),
                    line=line_number,
                    kind="guided_fix",
                    severity="medium",
                    summary="Se detectó una mejora guiada embebida en el código.",
                    evidence=line.strip(),
                    auto_fixable=True,
                    fix_hint=f'Reemplazar "{match.group("old")}" por "{match.group("new")}".',
                    metadata={"old": match.group("old"), "new": match.group("new")},
                )
            )
        return issues

    def _static_issues(self, file_path: Path, text: str, *, failure_history: list[str]) -> list[SelfImprovementIssue]:
        issues: list[SelfImprovementIssue] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "todo" in line.casefold() or "fixme" in line.casefold():
                issues.append(
                    SelfImprovementIssue(
                        issue_id=f"todo:{file_path}:{line_number}",
                        file_path=str(file_path),
                        line=line_number,
                        kind="todo_marker",
                        severity="low",
                        summary="Hay una marca TODO/FIXME pendiente.",
                        evidence=line.strip(),
                        auto_fixable=False,
                    )
                )
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            issues.append(
                SelfImprovementIssue(
                    issue_id=f"syntax:{file_path}:{exc.lineno}",
                    file_path=str(file_path),
                    line=exc.lineno,
                    kind="syntax_error",
                    severity="high",
                    summary="El archivo no compila correctamente.",
                    evidence=str(exc),
                    auto_fixable=False,
                )
            )
            return issues

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(
                    SelfImprovementIssue(
                        issue_id=f"bare-except:{file_path}:{node.lineno}",
                        file_path=str(file_path),
                        line=node.lineno,
                        kind="bare_except",
                        severity="medium",
                        summary="Se detectó un bare except.",
                        evidence="except:",
                        auto_fixable=False,
                    )
                )
        if any(str(file_path) in entry for entry in failure_history):
            issues.append(
                SelfImprovementIssue(
                    issue_id=f"failure-history:{file_path}",
                    file_path=str(file_path),
                    kind="failure_history",
                    severity="medium",
                    summary="El historial operativo contiene fallos relacionados con este archivo.",
                    evidence="\n".join(entry for entry in failure_history if str(file_path) in entry)[:500],
                    auto_fixable=False,
                )
            )
        return issues

    @staticmethod
    def _collect_python_files(target: Path) -> list[Path]:
        if target.is_file():
            return [target]
        return sorted(path for path in target.rglob("*.py") if "__pycache__" not in path.parts and ".pytest_cache" not in path.parts)

    def _collect_failure_history(self) -> list[str]:
        history: list[str] = []
        if self._ops_runtime is not None:
            try:
                snapshot = self._ops_runtime.snapshot()
                history.extend(str(item) for item in snapshot.recent_failures[:10])
            except Exception:  # noqa: BLE001
                pass
        if self._log_file is not None and self._log_file.exists():
            try:
                lines = self._log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                recent = [line for line in lines if '"level": "ERROR"' in line or '"recoverable": false' in line]
                history.extend(recent[-10:])
            except Exception:  # noqa: BLE001
                pass
        return history[:20]

    def _research_context(self) -> dict[str, object]:
        if self._research_runtime is None:
            return {}
        try:
            status = self._research_runtime.status()
        except Exception:  # noqa: BLE001
            return {}
        return {"research_status": status}
