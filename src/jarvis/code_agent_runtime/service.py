from __future__ import annotations

from pathlib import Path

from jarvis.code_agent_runtime.base import CodeTaskRequest
from jarvis.code_agent_runtime.executor import CodeAgentExecutor
from jarvis.code_agent_runtime.tool_router import CodeAgentToolRouter


class CodeAgentRuntimeService:
    def __init__(self, project_root: str | Path | None = None, *, runtime_dir: str | Path | None = None) -> None:
        root = Path(project_root or Path.cwd()).expanduser().resolve(strict=False)
        runtime = Path(runtime_dir).expanduser().resolve(strict=False) if runtime_dir else None
        self.executor = CodeAgentExecutor(root, runtime_dir=runtime)
        self.router = CodeAgentToolRouter(self.executor)

    def handle(self, request: CodeTaskRequest | dict):
        return self.router.route(CodeTaskRequest.model_validate(request))

    def scan_project(self):
        return self.executor.scan_project()

    def read_file(self, path: str):
        return self.executor.read_file(path)

    def search_content(self, query: str):
        return self.executor.search_project(query, mode="content")

    def search_name(self, query: str):
        return self.executor.search_project(query, mode="name")

    def write_file(self, path: str, content: str, *, overwrite: bool = False, confirm: bool = False, pin: str | None = None, dry_run: bool = False):
        return self.executor.write_file(path, content, overwrite=overwrite, confirm=confirm, pin=pin, dry_run=dry_run)

    def run_command(self, command: str, *, confirm: bool = False, pin: str | None = None, dry_run: bool = False):
        return self.executor.run_command(command, confirm=confirm, pin=pin, dry_run=dry_run)

    def configure_pin(self, pin: str):
        return self.executor.configure_pin(pin)

    def change_pin(self, current_pin: str, new_pin: str):
        return self.executor.change_pin(current_pin, new_pin)

    def current_mode(self):
        return self.executor.current_mode()

    def set_mode(self, mode: str):
        return self.executor.set_mode(mode)

    def action_log(self, limit: int = 20):
        return self.executor.log_tail(limit)

    def memory_show(self):
        return self.executor.memory().load()

    def memory_summary(self, max_chars: int = 4000):
        return self.executor.memory().get_agent_context_summary(max_chars=max_chars)

    def memory_add_note(self, text: str):
        return self.executor.memory().add_note(text)

    def memory_add_task(self, title: str, description: str = "", priority: str = "normal"):
        return self.executor.memory().add_pending_task(title, description=description, priority=priority)

    def memory_complete_task(self, task_id: str):
        return self.executor.memory().complete_task(task_id)

    def memory_scan_project(self):
        return self.executor.memory().scan_project()

    def memory_add_phase(self, name: str, notes: str = ""):
        return self.executor.memory().add_phase_completed(name, notes=notes)

    def skills_list(self):
        return self.executor.skills_list()

    def skills_show(self, skill_id: str):
        return self.executor.skills_show(skill_id)

    def skills_by_tag(self, tag: str):
        return self.executor.skills_by_tag(tag)

    def skills_suggest(self, task: str, *, limit: int = 5):
        return self.executor.skills_suggest(task, limit=limit)

    def skills_context(self, task: str, *, limit: int = 5, max_memory_chars: int = 2000):
        return self.executor.skills_context(task, limit=limit, max_memory_chars=max_memory_chars)

    def repos_index(self, library_root: str, *, max_repos: int | None = None):
        return self.executor.repos_index(library_root, max_repos=max_repos)

    def repos_list(self, *, limit: int = 100):
        return self.executor.repos_list(limit=limit)

    def repos_stats(self):
        return self.executor.repos_stats()

    def repos_show(self, repo_id: str):
        return self.executor.repos_show(repo_id)

    def repos_search(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None):
        return self.executor.repos_search(query, limit=limit, skill_ids=skill_ids)

    def repos_search_task(self, task: str, *, limit: int = 10):
        return self.executor.repos_search_task(task, limit=limit)

    def learn_search_github(self, query: str, *, max_results: int = 20, language: str | None = None, topic: str | None = None):
        return self.executor.learn_search_github(query, max_results=max_results, language=language, topic=topic)

    def learn_shortlist(self, task: str, *, max_results: int = 10, language: str | None = None, topic: str | None = None):
        return self.executor.learn_shortlist(task, max_results=max_results, language=language, topic=topic)

    def learn_clone(self, repo_id: str, *, library_root: str, confirm: bool = False, overwrite: bool = False):
        return self.executor.learn_clone(repo_id, library_root=library_root, confirm=confirm, overwrite=overwrite)

    def learn_clone_and_index(self, repo_id: str, *, library_root: str, confirm: bool = False, overwrite: bool = False):
        return self.executor.learn_clone_and_index(repo_id, library_root=library_root, confirm=confirm, overwrite=overwrite)

    def learn_extract(self):
        return self.executor.learn_extract()

    def learn_list(self, *, limit: int = 100):
        return self.executor.learn_list(limit=limit)

    def learn_search(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None):
        return self.executor.learn_search(query, limit=limit, skill_ids=skill_ids)

    def learn_for_task(self, task: str, *, limit: int = 8):
        return self.executor.learn_for_task(task, limit=limit)

    def learn_stats(self):
        return self.executor.learn_stats()

    def learn_summary(self, *, limit: int = 8):
        return self.executor.learn_summary(limit=limit)

    def local_search_rebuild(self):
        return self.executor.local_search_rebuild()

    def local_search_query(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None):
        return self.executor.local_search_query(query, limit=limit, skill_ids=skill_ids)

    def local_search_task(self, task: str, *, limit: int = 10):
        return self.executor.local_search_task(task, limit=limit)

    def local_search_context_for_task(self, task: str, *, max_results: int = 10, max_chars: int = 4000):
        return self.executor.local_search_context_for_task(task, max_results=max_results, max_chars=max_chars)

    def local_search_stats(self):
        return self.executor.local_search_stats()

    def local_search_clear(self, *, confirm: bool = False):
        return self.executor.local_search_clear(confirm=confirm)

    def agent_context(self, task: str):
        return self.executor.agent_context(task)

    def agent_plan(self, task: str, *, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3):
        return self.executor.agent_plan(task, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited)

    def agent_run(self, task: str, *, mode: str = "dry-run", confirm: bool = False, pin: str | None = None, patch_id: str | None = None, generate_patch: bool = False, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3):
        if patch_id:
            return self.executor.agent_run_patch(task, patch_id, mode=mode, confirm=confirm, pin=pin, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited)
        return self.executor.agent_run(task, mode=mode, confirm=confirm, pin=pin, generate_patch=generate_patch, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited)

    def agent_verify(self):
        return self.executor.agent_verify()

    def patch_propose_replace(self, file: str, old: str, new: str, *, task: str = "replace text"):
        return self.executor.patch_propose_replace(file, old, new, task=task)

    def patch_propose_insert_after(self, file: str, anchor: str, text: str, *, task: str = "insert text after anchor"):
        return self.executor.patch_propose_insert_after(file, anchor, text, task=task)

    def patch_propose_insert_before(self, file: str, anchor: str, text: str, *, task: str = "insert text before anchor"):
        return self.executor.patch_propose_insert_before(file, anchor, text, task=task)

    def patch_propose_append(self, file: str, text: str, *, task: str = "append text"):
        return self.executor.patch_propose_append(file, text, task=task)

    def patch_propose_create_file(self, file: str, content: str, *, task: str = "create file"):
        return self.executor.patch_propose_create_file(file, content, task=task)

    def patch_propose_unified_diff(self, diff_text: str, *, task: str = "unified diff"):
        return self.executor.patch_propose_unified_diff(diff_text, task=task)

    def patch_list(self, *, limit: int = 100):
        return self.executor.patch_list(limit=limit)

    def patch_show(self, patch_id: str):
        return self.executor.patch_show(patch_id)

    def patch_apply(self, patch_id: str, *, confirm: bool = False, pin: str | None = None, checkpoint: bool = False):
        return self.executor.patch_apply(patch_id, confirm=confirm, pin=pin, checkpoint=checkpoint)

    def patch_reject(self, patch_id: str):
        return self.executor.patch_reject(patch_id)

    def patch_stats(self):
        return self.executor.patch_stats()

    def change_targets(self, task: str, *, max_targets: int = 3):
        return self.executor.change_targets(task, max_targets=max_targets)

    def change_plan(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False):
        return self.executor.change_plan(task, max_targets=max_targets, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online)

    def change_propose(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False):
        return self.executor.change_propose(task, max_targets=max_targets, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online)

    def llm_status(self):
        return self.executor.llm_status()

    def llm_config(self):
        return self.executor.llm_config_view()

    def llm_mode(self):
        return self.executor.llm_mode_view()

    def llm_route(self, task: str, *, llm_mode: str | None = None, allow_online: bool = False):
        return self.executor.llm_route(task, llm_mode=llm_mode, allow_online=allow_online)

    def git_status(self):
        return self.executor.git_status()

    def git_current_branch(self):
        return self.executor.git_current_branch()

    def git_diff(self, path: str | None = None):
        return self.executor.git_diff(path=path)

    def git_diff_stat(self):
        return self.executor.git_diff_stat()

    def git_changed_files(self):
        return self.executor.git_changed_files()

    def git_summary(self):
        return self.executor.git_summary()

    def git_checkpoint(self, *, confirm: bool = False, pin: str | None = None, message: str | None = None):
        return self.executor.git_checkpoint(confirm=confirm, pin=pin, message=message)

    def git_create_branch(self, name: str, *, confirm: bool = False, pin: str | None = None):
        return self.executor.git_create_branch(name, confirm=confirm, pin=pin)

    def git_revert_file(self, path: str, *, confirm: bool = False, pin: str | None = None):
        return self.executor.git_revert_file(path, confirm=confirm, pin=pin)
