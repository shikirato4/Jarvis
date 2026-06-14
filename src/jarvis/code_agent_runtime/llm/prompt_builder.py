from __future__ import annotations

from typing import TYPE_CHECKING

from jarvis.code_agent_runtime.change_generator.models import ResolvedTarget
from jarvis.code_agent_runtime.llm.models import LLMGenerateRequest

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


class LLMPromptBuilder:
    max_prompt_chars = 8_000
    max_file_chars = 1_500
    max_targets = 3

    def __init__(self, executor: "CodeAgentExecutor") -> None:
        self._executor = executor

    def build(self, task: str, targets: list[ResolvedTarget], *, skills: list[str]) -> LLMGenerateRequest:
        safe_task = self._executor.project_memory.sanitize_text(task)
        safe_targets = [target for target in targets if not target.blocked][: self.max_targets]
        lines = [
            "You are generating a structured proposal for a reviewable Jarvis patch.",
            "Return only valid JSON. Do not include markdown fences.",
            "Allowed operation types: replace, insert_before, insert_after, append, create_file.",
            "Forbidden operations: delete_file, arbitrary_shell, unified_diff, install_dependency, modify_secret_file, edit_outside_project.",
            "Do not request command execution. Do not include secrets. Keep changes minimal.",
            f"Task: {safe_task}",
            f"Skills: {', '.join(skills[:8])}",
            f"Targets: {', '.join(target.path for target in safe_targets) or 'none'}",
            "Required JSON keys: status, summary, confidence, target_files, operations, warnings, tests_suggested.",
        ]
        memory_summary = self._executor.project_memory.get_agent_context_summary(max_chars=900)
        lines.append("Memory summary:")
        lines.append(memory_summary)
        search_context = self._safe_search_context(safe_task, skills)
        if search_context:
            lines.append("Local search context:")
            lines.append(search_context[:1200])
        file_sections = self._target_snippets(safe_targets)
        if file_sections:
            lines.append("Target file snippets:")
            lines.extend(file_sections)
        prompt = self._executor.project_memory.sanitize_text("\n".join(lines))[: self.max_prompt_chars]
        return LLMGenerateRequest(task=safe_task, prompt=prompt, target_files=[target.path for target in safe_targets], skills=skills)

    def _safe_search_context(self, task: str, skills: list[str]) -> str:
        try:
            result = self._executor.local_search.context_for_task(task, skill_ids=skills, max_results=4, max_chars=1200)
        except Exception:  # noqa: BLE001
            return ""
        return self._executor.project_memory.sanitize_text(str(result.get("context", "")))

    def _target_snippets(self, targets: list[ResolvedTarget]) -> list[str]:
        sections: list[str] = []
        for target in targets:
            if not target.exists:
                continue
            try:
                result = self._executor.reader.read(target.path)
            except Exception:  # noqa: BLE001
                continue
            snippet = self._executor.project_memory.sanitize_text(result.content[: self.max_file_chars])
            if snippet and snippet != "[redacted]":
                sections.append(f"FILE {result.path}:\n{snippet}")
        return sections[: self.max_targets]
