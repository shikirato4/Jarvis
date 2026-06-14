from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.code_agent_runtime.change_generator.models import ResolvedTarget
from jarvis.code_agent_runtime.paths import is_sensitive_path, looks_like_text_path, normalize_project_path, relative_to_root

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:[A-Za-z]:[\\/][^\s\"'`]+)"
    r"|(?:\.\.?[\\/])?[^\s\"'`]+[\\/][^\s\"'`]+\.\w+"
    r"|(?:README\.md|\.env(?:\.\w+)?|[^\s\"'`\\/]+\.(?:py|ts|tsx|js|jsx|md|txt|json|toml|yaml|yml|html|css))"
    r")",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"'`](?P<value>[^\"'`]+)[\"'`]")


class TargetResolver:
    def __init__(self, executor: "CodeAgentExecutor", *, max_targets: int = 3) -> None:
        self._executor = executor
        self._root = executor.project_root
        self._max_targets = max_targets

    def resolve(self, task: str, *, max_targets: int | None = None) -> tuple[list[ResolvedTarget], list[str], list[str]]:
        limit = max_targets or self._max_targets
        reasons: list[str] = []
        warnings: list[str] = []
        candidates = self._explicit_paths(task)
        if self._mentions_readme(task):
            candidates.insert(0, "README.md")
        targets: list[ResolvedTarget] = []
        for candidate in candidates:
            target = self._target_from_path(candidate, "explicit path mentioned in task")
            targets.append(target)
            if target.blocked:
                warnings.append(target.blocked_reason)
        if targets:
            return self._dedupe(targets)[:limit], reasons, warnings

        context = self._safe_context(task)
        for path in self._paths_from_context(context):
            target = self._target_from_path(path, "suggested by local context", confidence=0.45)
            if not target.blocked:
                targets.append(target)
            if len(targets) >= limit:
                break
        if targets:
            reasons.append("targets were inferred from project context and require review")
            return self._dedupe(targets), reasons, warnings
        reasons.append("no clear editable target was found")
        return [], reasons, warnings

    def _target_from_path(self, raw_path: str, reason: str, *, confidence: float = 0.85) -> ResolvedTarget:
        cleaned = self._clean_path(raw_path)
        try:
            target = normalize_project_path(self._root, cleaned)
        except Exception as exc:  # noqa: BLE001
            return ResolvedTarget(path=cleaned, reason=reason, confidence=0.0, blocked=True, blocked_reason=str(exc))
        if is_sensitive_path(target):
            return ResolvedTarget(path=cleaned, reason=reason, confidence=0.0, exists=target.exists(), sensitive=True, blocked=True, blocked_reason=f"sensitive file is blocked: {cleaned}")
        if self._executor.path_policy.is_protected_project_file(target):
            return ResolvedTarget(path=relative_to_root(self._root, target), reason=reason, confidence=0.0, exists=target.exists(), sensitive=False, blocked=True, blocked_reason=f"protected project file requires explicit patch operation: {cleaned}")
        if target.exists() and not looks_like_text_path(target):
            return ResolvedTarget(path=relative_to_root(self._root, target), reason=reason, confidence=0.0, exists=True, blocked=True, blocked_reason=f"non-text file is blocked: {cleaned}")
        return ResolvedTarget(path=relative_to_root(self._root, target), reason=reason, confidence=confidence, exists=target.exists(), sensitive=False, blocked=False)

    def _safe_context(self, task: str) -> dict[str, Any]:
        try:
            return self._executor.local_search.context_for_task(task, skill_ids=[], max_results=6, max_chars=1200)
        except Exception:  # noqa: BLE001
            return {}

    def _paths_from_context(self, context: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for result in context.get("results", []):
            if isinstance(result, dict):
                path = result.get("path") or result.get("file") or result.get("source_file")
                if path:
                    paths.append(str(path))
        return paths

    def _explicit_paths(self, task: str) -> list[str]:
        paths: list[str] = []
        for match in _QUOTED_RE.finditer(task):
            value = match.group("value").strip()
            if self._looks_like_path(value):
                paths.append(value)
        for match in _PATH_RE.finditer(task):
            paths.append(match.group("path"))
        return self._dedupe_strings(paths)

    @staticmethod
    def _mentions_readme(task: str) -> bool:
        folded = task.casefold()
        return "readme" in folded and any(token in folded for token in ("agrega", "añade", "append", "nota", "note", "document"))

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        if "\\" in value or "/" in value:
            return True
        return bool(_PATH_RE.fullmatch(value.strip()))

    @staticmethod
    def _clean_path(value: str) -> str:
        cleaned = value.strip().strip(" ,;:()[]{}<>\"'`").replace("\\", "/")
        if cleaned.endswith(".") and not cleaned.casefold().endswith((".env", ".env.local", ".env.production")):
            cleaned = cleaned[:-1]
        return cleaned

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in values if item))

    @staticmethod
    def _dedupe(values: list[ResolvedTarget]) -> list[ResolvedTarget]:
        seen: set[str] = set()
        result: list[ResolvedTarget] = []
        for value in values:
            key = value.path.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(value)
        return result
