from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from jarvis.code_agent_runtime.change_generator.models import ChangeOperation, GeneratedChangePlan, ResolvedTarget
from jarvis.code_agent_runtime.llm.models import LLMChangeOperation, LLMChangeProposal
from jarvis.code_agent_runtime.paths import is_sensitive_path, normalize_project_path, relative_to_root
from jarvis.code_agent_runtime.security.path_policy import PathPolicy


ALLOWED_OPERATIONS = {"replace", "insert_before", "insert_after", "append", "create_file"}
DANGEROUS_SHELL_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bdel\s+/s\b",
    r"\brmdir\s+/s\b",
    r"\bsudo\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-f",
    r"\bcurl\b.*\|\s*(?:sh|bash)",
    r"\bwget\b.*\|\s*(?:sh|bash)",
)


class LLMResponseParser:
    max_content_chars = 20_000
    min_confidence = 0.4

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve(strict=False)
        self._path_policy = PathPolicy(self._root)

    def parse(self, raw: str, *, task: str, skills: list[str]) -> GeneratedChangePlan:
        try:
            payload = json.loads(self._extract_json(raw))
        except json.JSONDecodeError as exc:
            return self._needs_review(task, skills, f"LLM returned invalid JSON: {exc.msg}")
        if not isinstance(payload, dict):
            return self._needs_review(task, skills, "LLM response JSON must be an object")
        try:
            proposal = LLMChangeProposal.model_validate(payload | {"raw": {}})
        except Exception as exc:  # noqa: BLE001
            return self._needs_review(task, skills, f"LLM response schema is invalid: {exc}")
        if proposal.confidence < self.min_confidence:
            return self._needs_review(task, skills, "LLM confidence is too low")
        operations: list[ChangeOperation] = []
        targets: list[ResolvedTarget] = []
        warnings = list(proposal.warnings)
        for operation in proposal.operations:
            validated = self._validate_operation(operation)
            if isinstance(validated, str):
                return self._blocked(task, skills, validated)
            change, target = validated
            operations.append(change)
            targets.append(target)
        if not operations:
            return self._needs_review(task, skills, "LLM did not propose any allowed operation")
        return GeneratedChangePlan(
            task=task,
            status="proposed" if proposal.status == "proposed" else "needs_review",
            skills=skills,
            targets=targets,
            operations=operations,
            confidence=proposal.confidence,
            risks=["LLM proposal converted to explicit reviewable patch operations"],
            warnings=warnings,
            reasons=[proposal.summary or "LLM structured proposal"],
            context_used=["llm_provider", "prompt_builder", "response_parser"],
        )

    def _validate_operation(self, operation: LLMChangeOperation) -> tuple[ChangeOperation, ResolvedTarget] | str:
        op_type = operation.type.strip().casefold()
        if op_type not in ALLOWED_OPERATIONS:
            return f"LLM proposed forbidden operation: {op_type}"
        if not operation.file:
            return "LLM operation is missing file"
        try:
            target = normalize_project_path(self._root, operation.file)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        rel = relative_to_root(self._root, target)
        if is_sensitive_path(target) or self._path_policy.is_sensitive(target):
            return f"LLM target is sensitive and blocked: {rel}"
        if self._path_policy.is_protected_project_file(target):
            return f"LLM target is protected and blocked: {rel}"
        if op_type != "create_file" and not target.exists():
            return f"LLM target file does not exist: {rel}"
        if len(operation.old_text + operation.new_text + operation.anchor + operation.text + operation.content) > self.max_content_chars:
            return "LLM operation content is too large"
        if self._contains_dangerous_shell(operation.old_text, operation.new_text, operation.anchor, operation.text, operation.content, operation.reason):
            return "LLM operation contains dangerous shell content"
        if op_type == "replace" and (not operation.old_text or not operation.new_text):
            return "replace operation requires old_text and new_text"
        if op_type in {"insert_before", "insert_after"} and (not operation.anchor or not operation.text):
            return f"{op_type} operation requires anchor and text"
        if op_type == "append" and not operation.text:
            return "append operation requires text"
        if op_type == "create_file" and not operation.content:
            return "create_file operation requires content"
        change = ChangeOperation(
            operation=op_type,
            file=rel,
            old_text=operation.old_text,
            new_text=operation.new_text,
            anchor=operation.anchor,
            text=operation.text,
            content=operation.content,
            reason=operation.reason or "LLM structured operation",
        )
        target_info = ResolvedTarget(path=rel, reason="LLM structured proposal", confidence=0.7, exists=target.exists())
        return change, target_info

    @staticmethod
    def _extract_json(raw: str) -> str:
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.casefold().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return stripped[start : end + 1]
        return stripped

    @staticmethod
    def _contains_dangerous_shell(*values: str) -> bool:
        blob = "\n".join(values)
        return any(re.search(pattern, blob, flags=re.IGNORECASE) for pattern in DANGEROUS_SHELL_PATTERNS)

    @staticmethod
    def _needs_review(task: str, skills: list[str], reason: str) -> GeneratedChangePlan:
        return GeneratedChangePlan(task=task, status="needs_review", skills=skills, confidence=0.2, warnings=["no patch was generated"], reasons=[reason], context_used=["llm_response_parser"])

    @staticmethod
    def _blocked(task: str, skills: list[str], reason: str) -> GeneratedChangePlan:
        return GeneratedChangePlan(task=task, status="blocked", skills=skills, confidence=0.0, risks=[reason], warnings=["LLM proposal was blocked"], reasons=[reason], context_used=["llm_response_parser"])
