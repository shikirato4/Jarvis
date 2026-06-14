from __future__ import annotations

from pathlib import Path

from jarvis.code_agent_runtime.base import CodeActionKind, RiskAssessment, RiskLevel
from jarvis.code_agent_runtime.paths import is_inside_project
from jarvis.code_agent_runtime.security.command_validator import CommandValidator
from jarvis.code_agent_runtime.security.path_policy import PathPolicy


class RiskClassifier:
    def __init__(self) -> None:
        self._command_validator = CommandValidator()

    def assess_file_action(self, action: CodeActionKind, project_root: Path, path: Path, *, exists: bool = False, overwrite: bool = False) -> RiskAssessment:
        path_policy = PathPolicy(project_root)
        if not is_inside_project(project_root, path):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="path is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])
        if path_policy.is_sensitive(path):
            return RiskAssessment(level=RiskLevel.CRITICAL, reason="sensitive file requires special approval", requires_confirmation=True, requires_pin=True, tags=["sensitive_path"])
        if action in {CodeActionKind.FILE_READ, CodeActionKind.PROJECT_SEARCH}:
            return RiskAssessment(level=RiskLevel.SAFE, reason="read-only project action")
        if path_policy.is_protected_project_file(path):
            return RiskAssessment(level=RiskLevel.SENSITIVE, reason="protected project configuration file requires confirmation and PIN", requires_confirmation=True, requires_pin=True, tags=["protected_project_file"])
        if overwrite or exists:
            return RiskAssessment(level=RiskLevel.SENSITIVE, reason="existing file overwrite requires confirmation", requires_confirmation=True, requires_pin=False, tags=["overwrite"])
        return RiskAssessment(level=RiskLevel.MINOR_CHANGE, reason="new normal project file")

    def assess_scan_or_search(self) -> RiskAssessment:
        return RiskAssessment(level=RiskLevel.SAFE, reason="read-only project discovery")

    def assess_command(self, command: str, project_root: Path, cwd: Path | None = None) -> RiskAssessment:
        return self._command_validator.validate(command, project_root, cwd=cwd)

    def command_argv(self, command: str) -> list[str]:
        return self._command_validator.to_argv(command)
