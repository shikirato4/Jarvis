from __future__ import annotations

import re
from typing import TYPE_CHECKING

from jarvis.code_agent_runtime.change_generator.models import ChangeOperation, GeneratedChangePlan, ResolvedTarget

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


_BROAD_TERMS = (
    "arregla todo",
    "refactoriza toda",
    "refactoriza todo",
    "reescribe todo",
    "rewrite everything",
    "fix everything",
)


class ChangePlanner:
    def __init__(self, executor: "CodeAgentExecutor") -> None:
        self._executor = executor

    def plan(self, task: str, targets: list[ResolvedTarget], *, skills: list[str], context_used: list[str]) -> GeneratedChangePlan:
        safe_task = self._executor.project_memory.sanitize_text(task)
        blocked_targets = [target for target in targets if target.blocked]
        if blocked_targets:
            return GeneratedChangePlan(
                task=safe_task,
                status="blocked",
                skills=skills,
                targets=targets,
                risks=[target.blocked_reason for target in blocked_targets],
                warnings=["one or more targets are blocked"],
                reasons=[target.blocked_reason for target in blocked_targets],
                context_used=context_used,
            )
        folded = safe_task.casefold()
        if any(term in folded for term in _BROAD_TERMS):
            return self._needs_review(safe_task, targets, skills, context_used, "task scope is too broad for deterministic patch generation")
        if not targets:
            return self._needs_review(safe_task, targets, skills, context_used, "no clear target file was resolved")
        if len(targets) > 1 and not self._allows_multi_target(folded):
            return self._needs_review(safe_task, targets, skills, context_used, "multiple possible targets require user review")

        operation = self._replace_operation(safe_task, targets[0])
        if operation is None:
            operation = self._create_file_operation(safe_task, targets[0])
        if operation is None:
            operation = self._append_operation(safe_task, targets[0])
        if operation is None:
            return self._needs_review(safe_task, targets, skills, context_used, "no safe deterministic edit pattern matched the task")

        return GeneratedChangePlan(
            task=safe_task,
            status="proposed",
            skills=skills,
            targets=targets[:1],
            operations=[operation],
            confidence=0.8,
            risks=["patch is reviewable and is not applied automatically"],
            warnings=["review the generated diff before applying"],
            reasons=[operation.reason],
            context_used=context_used,
        )

    def _replace_operation(self, task: str, target: ResolvedTarget) -> ChangeOperation | None:
        quoted = re.search(r"(?:cambia|reemplaza|replace)\s+[\"'`](?P<old>.+?)[\"'`]\s+(?:por|with)\s+[\"'`](?P<new>.*?)[\"'`]", task, flags=re.IGNORECASE)
        if quoted:
            return ChangeOperation(operation="replace", file=target.path, old_text=quoted.group("old"), new_text=quoted.group("new"), reason="exact quoted replace requested")
        simple = re.search(r"(?:cambia|reemplaza|replace)\s+(?P<old>\S+)\s+(?:por|with)\s+(?P<new>\S+)", task, flags=re.IGNORECASE)
        if simple:
            return ChangeOperation(operation="replace", file=target.path, old_text=simple.group("old"), new_text=simple.group("new"), reason="exact replace requested")
        return None

    def _create_file_operation(self, task: str, target: ResolvedTarget) -> ChangeOperation | None:
        folded = task.casefold()
        if not any(token in folded for token in ("crea", "crear", "create")):
            return None
        if target.exists:
            return None
        content = self._extract_content(task)
        if not content:
            title = target.path.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", " ").replace("_", " ").title()
            content = f"# {title}\n\nCreated as a reviewable Jarvis patch proposal.\n"
        return ChangeOperation(operation="create_file", file=target.path, content=self._ensure_newline(content), reason="explicit create-file task with a clear target")

    def _append_operation(self, task: str, target: ResolvedTarget) -> ChangeOperation | None:
        folded = task.casefold()
        if not any(token in folded for token in ("agrega", "añade", "append", "add", "nota", "note")):
            return None
        text = self._extract_content(task)
        if not text:
            text = f"> Nota: {task}"
        return ChangeOperation(operation="append", file=target.path, text="\n\n" + self._ensure_newline(text), reason="small append requested for a clear target")

    @staticmethod
    def _extract_content(task: str) -> str:
        match = re.search(r"(?:con contenido|with content|con texto|with text|con)\s+[\"'`](?P<content>.+?)[\"'`]\s*$", task, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group("content").strip()
        return ""

    @staticmethod
    def _ensure_newline(value: str) -> str:
        return value if value.endswith("\n") else f"{value}\n"

    @staticmethod
    def _allows_multi_target(task: str) -> bool:
        return "todos estos archivos" in task or "multiple files" in task

    @staticmethod
    def _needs_review(task: str, targets: list[ResolvedTarget], skills: list[str], context_used: list[str], reason: str) -> GeneratedChangePlan:
        return GeneratedChangePlan(
            task=task,
            status="needs_review",
            skills=skills,
            targets=targets,
            confidence=0.25,
            warnings=["no patch was generated"],
            reasons=[reason],
            context_used=context_used,
        )
