from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.code_agent_runtime.agent.models import AgentContext

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


class AgentContextBuilder:
    def __init__(self, executor: "CodeAgentExecutor", *, max_chars: int = 6000) -> None:
        self._executor = executor
        self._max_chars = max_chars

    def build(self, task: str, *, skill_limit: int = 5) -> AgentContext:
        safe_task = self._executor.project_memory.sanitize_text(task)
        skills_payload = self._executor.skills_context(safe_task, limit=skill_limit, max_memory_chars=1500)
        skill_ids = []
        for item in skills_payload.get("suggested_skills", []):
            if isinstance(item, dict) and item.get("id"):
                skill_ids.append(str(item["id"]))
            elif isinstance(item, dict) and isinstance(item.get("skill"), dict) and item["skill"].get("id"):
                skill_ids.append(str(item["skill"]["id"]))
        search_context = self._executor.local_search_context_for_task(safe_task, max_results=6, max_chars=1600)
        learning_results = self._executor.learn_for_task(safe_task, limit=4)
        repo_results = self._executor.repos_search_task(safe_task, limit=4)
        scan_receipt = self._executor.scan_project()
        project_structure = scan_receipt.data.get("scan", {}) if scan_receipt.status.value == "ok" else {}
        git_receipt = self._executor.git_summary()
        git_summary = git_receipt.data.get("git", {}) if git_receipt.status.value == "ok" else {"status": git_receipt.status.value, "message": git_receipt.message}
        relevant_files = self._relevant_files(project_structure, repo_results, learning_results)
        context = AgentContext(
            task=safe_task,
            memory_summary=skills_payload.get("memory_summary", ""),
            skills=skills_payload.get("suggested_skills", []),
            skill_contexts=skills_payload.get("skill_contexts", []),
            git_summary=git_summary,
            search_context=search_context,
            learning_results=self._strip_results(learning_results),
            repo_results=self._strip_results(repo_results),
            project_structure=self._strip_project_structure(project_structure),
            relevant_files=relevant_files,
            warnings=[*search_context.get("warnings", []), *learning_results.get("warnings", []), *repo_results.get("warnings", [])],
        )
        context.context_summary = self._summary(context)
        return AgentContext.model_validate(self._executor.project_memory.sanitize_value(context.model_dump(mode="json")))

    def _summary(self, context: AgentContext) -> str:
        lines = [
            f"Task: {context.task}",
            f"Skills: {', '.join(self._skill_ids(context.skills))}",
            f"Memory: {context.memory_summary[:1200]}",
            f"Git: {context.git_summary}",
            f"Relevant files: {', '.join(context.relevant_files[:12])}",
            f"Local search: {context.search_context.get('context', '')[:1600]}",
        ]
        return self._executor.project_memory.sanitize_text("\n".join(lines))[: self._max_chars]

    @staticmethod
    def _skill_ids(skills: list[dict[str, Any]]) -> list[str]:
        ids: list[str] = []
        for item in skills:
            if not isinstance(item, dict):
                continue
            if item.get("id"):
                ids.append(str(item["id"]))
            elif isinstance(item.get("skill"), dict) and item["skill"].get("id"):
                ids.append(str(item["skill"]["id"]))
        return list(dict.fromkeys(ids))

    @staticmethod
    def _strip_results(payload: dict[str, Any]) -> dict[str, Any]:
        results = payload.get("results", []) if isinstance(payload, dict) else []
        compact = []
        for item in results[:4]:
            if isinstance(item, dict):
                compact.append({key: item.get(key) for key in ("title", "repo_id", "source_file", "file", "score", "match_reason", "skills", "notice") if key in item})
        return {"result_count": payload.get("result_count", len(compact)) if isinstance(payload, dict) else len(compact), "results": compact, "notice": payload.get("notice", "") if isinstance(payload, dict) else ""}

    @staticmethod
    def _strip_project_structure(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "root": payload.get("root", ""),
            "files": payload.get("files", [])[:30],
            "directories": payload.get("directories", [])[:30],
            "extension_counts": payload.get("extension_counts", {}),
            "summary": payload.get("summary", ""),
        }

    @staticmethod
    def _relevant_files(project_structure: dict[str, Any], repo_results: dict[str, Any], learning_results: dict[str, Any]) -> list[str]:
        files: list[str] = []
        for item in project_structure.get("files", [])[:20]:
            if isinstance(item, dict) and item.get("path"):
                files.append(str(item["path"]))
        for result in repo_results.get("results", [])[:5]:
            if isinstance(result, dict) and result.get("file"):
                files.append(str(result["file"]))
        for result in learning_results.get("results", [])[:5]:
            if isinstance(result, dict) and result.get("source_file"):
                files.append(str(result["source_file"]))
        return list(dict.fromkeys(files))[:20]
