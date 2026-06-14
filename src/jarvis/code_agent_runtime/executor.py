from __future__ import annotations

from pathlib import Path

from jarvis.code_agent_runtime.base import (
    AuthorizationContext,
    CodeActionKind,
    CodeActionStatus,
    CodeAgentReceipt,
    OperationMode,
    PermissionDecision,
    PermissionResult,
    RiskAssessment,
    RiskLevel,
    utcnow,
)
from jarvis.code_agent_runtime.agent import AgentRunMode, AgentRunner
from jarvis.code_agent_runtime.change_generator import ChangeGenerator
from jarvis.code_agent_runtime.llm import LLMConfig, LLMRouter, build_llm_provider
from jarvis.code_agent_runtime.memory.project_memory import ProjectMemory
from jarvis.code_agent_runtime.patches import PatchApplier, PatchBuilder, PatchStore
from jarvis.code_agent_runtime.git import GitManager
from jarvis.code_agent_runtime.paths import normalize_project_path
from jarvis.code_agent_runtime.repo_library import RepoLibraryIndex, RepoLibraryStorage
from jarvis.code_agent_runtime.repo_learning import GitHubDiscovery, LearningStorage, PatternStore, RepoLearningRouter
from jarvis.code_agent_runtime.search import LocalSearchService, SearchStorage
from jarvis.code_agent_runtime.security.action_log import ActionLog
from jarvis.code_agent_runtime.security.modes import OperationModeStore
from jarvis.code_agent_runtime.security.path_policy import PathPolicy
from jarvis.code_agent_runtime.security.permissions import PermissionGate
from jarvis.code_agent_runtime.security.pin_auth import PinAuthProvider
from jarvis.code_agent_runtime.security.risk_classifier import RiskClassifier
from jarvis.code_agent_runtime.skills import SkillRouter, build_builtin_registry
from jarvis.code_agent_runtime.tools.file_reader import FileReader
from jarvis.code_agent_runtime.tools.file_writer import FileWriter
from jarvis.code_agent_runtime.tools.project_scanner import ProjectScanner
from jarvis.code_agent_runtime.tools.project_search import ProjectSearch
from jarvis.code_agent_runtime.tools.terminal_runner import TerminalRunner
from jarvis.code_agent_runtime.verifier import CodeAgentVerifier


class CodeAgentExecutor:
    def __init__(
        self,
        project_root: Path,
        *,
        runtime_dir: Path | None = None,
        max_read_bytes: int = 120_000,
    ) -> None:
        self.project_root = project_root.resolve(strict=False)
        self.runtime_dir = (runtime_dir or self.project_root / "runtime" / "code_agent").resolve(strict=False)
        self.risk_classifier = RiskClassifier()
        self.mode_store = OperationModeStore(self.runtime_dir / "mode.json")
        self.path_policy = PathPolicy(self.project_root)
        self.pin_auth = PinAuthProvider(self.runtime_dir / "pin_auth.json")
        self.permission_gate = PermissionGate(self.pin_auth, self.path_policy)
        self.action_log = ActionLog(self.runtime_dir / "actions.jsonl")
        self.project_memory = ProjectMemory(self.runtime_dir / "project_memory.json", project_root=self.project_root)
        self.scanner = ProjectScanner(self.project_root)
        self.reader = FileReader(self.project_root, max_bytes=max_read_bytes)
        self.search = ProjectSearch(self.project_root, max_file_bytes=max_read_bytes)
        self.writer = FileWriter(self.project_root)
        self.runner = TerminalRunner(self.project_root)
        self.git = GitManager(self.project_root, self.runner)
        self.skill_registry = build_builtin_registry()
        self.skill_router = SkillRouter(self.skill_registry)
        self.repo_library = RepoLibraryIndex(RepoLibraryStorage(self.runtime_dir / "repo_library_index.json"))
        self.repo_learning = RepoLearningRouter(
            discovery=GitHubDiscovery(),
            repo_library=self.repo_library,
            pattern_store=PatternStore(LearningStorage(self.runtime_dir / "repo_learning_knowledge.json")),
        )
        self.local_search = LocalSearchService(
            storage=SearchStorage(self.runtime_dir / "search_index.sqlite"),
            repo_library=self.repo_library,
            repo_learning=self.repo_learning,
        )
        self.llm_config = LLMConfig.from_env_with_autodetect(project_root=self.project_root)
        self.llm_provider = build_llm_provider(self.llm_config)
        self.llm_router = LLMRouter(self.llm_config)
        self.patch_store = PatchStore(self.runtime_dir / "patches")
        self.patch_builder = PatchBuilder(self.project_root, self.patch_store)
        self.patch_applier = PatchApplier(self, self.patch_store)
        self.change_generator = ChangeGenerator(self)
        self.agent = AgentRunner(self)
        self.verifier = CodeAgentVerifier()

    def scan_project(self) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        risk = self.risk_classifier.assess_scan_or_search()
        try:
            result = self.scanner.scan()
            self.project_memory.update_project_structure(result.model_dump(mode="json"))
            receipt = CodeAgentReceipt(
                action=CodeActionKind.PROJECT_SCAN,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=mode,
                message=result.summary,
                target=str(self.project_root),
                tool="project_scanner",
                data={"scan": result.model_dump(mode="json")},
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.PROJECT_SCAN, risk, str(self.project_root), started, exc, mode=mode, tool="project_scanner")
        self.action_log.append(receipt)
        return receipt

    def read_file(self, path: str) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        try:
            target = normalize_project_path(self.project_root, path)
        except Exception as exc:  # noqa: BLE001
            risk = RiskAssessment(level=RiskLevel.CRITICAL, reason="path is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])
            receipt = CodeAgentReceipt(
                action=CodeActionKind.FILE_READ,
                status=CodeActionStatus.BLOCKED,
                risk=risk,
                mode=mode,
                message=str(exc),
                target=path,
                tool="file_reader",
                confirmation_required=True,
                pin_required=True,
                blocked_reason=str(exc),
                started_at=started,
                finished_at=utcnow(),
            )
            self.action_log.append(receipt)
            return receipt
        risk = self.risk_classifier.assess_file_action(CodeActionKind.FILE_READ, self.project_root, target)
        allowed = self._check_permission(risk, action=CodeActionKind.FILE_READ, target=str(target), started=started, mode=mode, path=target, tool="file_reader")
        if allowed is not None:
            return allowed
        try:
            result = self.reader.read(str(target))
            receipt = CodeAgentReceipt(
                action=CodeActionKind.FILE_READ,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=mode,
                message=f"read {result.path}",
                target=result.path,
                tool="file_reader",
                data={"file": result.model_dump(mode="json")},
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.FILE_READ, risk, str(target), started, exc, mode=mode, tool="file_reader")
        self.action_log.append(receipt)
        return receipt

    def search_project(self, query: str, *, mode: str = "content") -> CodeAgentReceipt:
        started = utcnow()
        active_mode = self.current_mode()
        risk = self.risk_classifier.assess_scan_or_search()
        try:
            result = self.search.search_name(query) if mode == "name" else self.search.search_content(query)
            receipt = CodeAgentReceipt(
                action=CodeActionKind.PROJECT_SEARCH,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=active_mode,
                message=f"found {len(result.matches)} matches",
                target=str(self.project_root),
                tool="project_search",
                data={"search": result.model_dump(mode="json")},
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.PROJECT_SEARCH, risk, str(self.project_root), started, exc, mode=active_mode, tool="project_search")
        self.action_log.append(receipt)
        return receipt

    def write_file(self, path: str, content: str, *, overwrite: bool = False, confirm: bool = False, pin: str | None = None, dry_run: bool = False) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        try:
            target = normalize_project_path(self.project_root, path)
        except Exception as exc:  # noqa: BLE001
            risk = RiskAssessment(level=RiskLevel.CRITICAL, reason="path is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])
            receipt = CodeAgentReceipt(
                action=CodeActionKind.FILE_WRITE,
                status=CodeActionStatus.BLOCKED,
                risk=risk,
                mode=mode,
                message=str(exc),
                target=path,
                tool="file_writer",
                confirmation_required=True,
                pin_required=True,
                blocked_reason=str(exc),
                started_at=started,
                finished_at=utcnow(),
            )
            self.action_log.append(receipt)
            return receipt
        risk = self.risk_classifier.assess_file_action(
            CodeActionKind.FILE_WRITE,
            self.project_root,
            target,
            exists=target.exists(),
            overwrite=overwrite and target.exists(),
        )
        blocked = self._check_permission(risk, action=CodeActionKind.FILE_WRITE, target=str(target), started=started, confirm=confirm, pin=pin, mode=mode, path=target, tool="file_writer")
        if blocked is not None:
            return blocked
        try:
            authorization = self._authorization_context(
                action=CodeActionKind.FILE_WRITE,
                target=str(target),
                risk=risk,
                mode=mode,
                confirm=confirm,
            )
            data = self.writer.write_text(str(target), content, overwrite=overwrite, dry_run=dry_run, authorization=authorization)
            if not dry_run:
                self.project_memory.add_note(f"Modified file: {data['path']}")
            receipt = CodeAgentReceipt(
                action=CodeActionKind.FILE_WRITE,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=mode,
                message=f"wrote {data['path']}",
                target=str(data["path"]),
                tool="file_writer",
                touched_files=[str(data["path"])],
                data={"write": data},
                pin_verified=True if risk.requires_pin else None,
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.FILE_WRITE, risk, str(target), started, exc, mode=mode, tool="file_writer")
        self.action_log.append(receipt)
        return receipt

    def run_command(self, command: str, *, confirm: bool = False, pin: str | None = None, dry_run: bool = False) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        risk = self.risk_classifier.assess_command(command, self.project_root)
        blocked = self._check_permission(risk, action=CodeActionKind.COMMAND_RUN, target=command, started=started, confirm=confirm, pin=pin, mode=mode, command=command, tool="terminal_runner")
        if blocked is not None:
            return blocked
        try:
            argv = self.risk_classifier.command_argv(command)
            authorization = self._authorization_context(
                action=CodeActionKind.COMMAND_RUN,
                target=command,
                risk=risk,
                mode=mode,
                confirm=confirm,
            )
            result = self.runner.run(command, argv=argv, dry_run=dry_run, authorization=authorization)
            success = self.verifier.command_succeeded(result.return_code)
            if success:
                self.project_memory.add_useful_command(command, "command completed", int(risk.level))
            else:
                self.project_memory.add_failed_command(command, result.stderr or f"exit code {result.return_code}", int(risk.level))
            receipt = CodeAgentReceipt(
                action=CodeActionKind.COMMAND_RUN,
                status=CodeActionStatus.OK if success else CodeActionStatus.FAILED,
                risk=risk,
                mode=mode,
                message="command completed" if success else f"command failed with exit code {result.return_code}",
                target=str(self.project_root),
                tool="terminal_runner",
                commands=[command],
                data={"command": result.model_dump(mode="json")},
                errors=[result.stderr] if result.stderr and not success else [],
                pin_verified=True if risk.requires_pin else None,
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.COMMAND_RUN, risk, command, started, exc, mode=mode, tool="terminal_runner")
        self.action_log.append(receipt)
        return receipt

    def configure_pin(self, pin: str) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        risk = RiskAssessment(level=RiskLevel.SENSITIVE, reason="master PIN configuration", requires_confirmation=True, requires_pin=False)
        try:
            self.pin_auth.configure_pin(pin)
            receipt = CodeAgentReceipt(
                action=CodeActionKind.PIN_CONFIGURE,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=mode,
                message="master PIN configured",
                target=str(self.runtime_dir / "pin_auth.json"),
                tool="pin_auth",
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            receipt = self._failure(CodeActionKind.PIN_CONFIGURE, risk, str(self.runtime_dir / "pin_auth.json"), started, exc, mode=mode, tool="pin_auth")
        self.action_log.append(receipt)
        return receipt

    def change_pin(self, current_pin: str, new_pin: str) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        risk = RiskAssessment(level=RiskLevel.SENSITIVE, reason="master PIN change", requires_confirmation=True, requires_pin=True)
        result = self.pin_auth.change_pin(current_pin, new_pin)
        receipt = CodeAgentReceipt(
            action=CodeActionKind.PIN_CHANGE,
            status=CodeActionStatus.OK if result.success else CodeActionStatus.BLOCKED,
            risk=risk,
            mode=mode,
            message=result.reason,
            target=str(self.runtime_dir / "pin_auth.json"),
            tool="pin_auth",
            pin_required=True,
            pin_verified=result.success,
            blocked_reason=None if result.success else result.reason,
            started_at=started,
            finished_at=utcnow(),
        )
        self.action_log.append(receipt)
        return receipt

    def current_mode(self) -> OperationMode:
        return self.mode_store.current()

    def set_mode(self, mode: OperationMode | str) -> CodeAgentReceipt:
        started = utcnow()
        selected = self.mode_store.set(mode)
        risk = RiskAssessment(level=RiskLevel.SAFE, reason="operation mode changed")
        receipt = CodeAgentReceipt(
            action=CodeActionKind.MODE_CHANGE,
            status=CodeActionStatus.OK,
            risk=risk,
            mode=selected,
            message=f"mode set to {selected.value}",
            target=str(self.runtime_dir / "mode.json"),
            tool="mode_store",
            data={"mode": selected.value},
            started_at=started,
            finished_at=utcnow(),
        )
        self.action_log.append(receipt)
        return receipt

    def log_tail(self, limit: int = 20) -> list[dict]:
        return self.action_log.tail(limit)

    def memory(self) -> ProjectMemory:
        return self.project_memory

    def skills_list(self) -> list[dict]:
        return [skill.to_dict() for skill in self.skill_registry.list()]

    def skills_show(self, skill_id: str) -> dict:
        return self.skill_registry.get(skill_id).to_dict()

    def skills_by_tag(self, tag: str) -> list[dict]:
        return [skill.to_dict() for skill in self.skill_registry.by_tag(tag)]

    def skills_suggest(self, task: str, *, limit: int = 5) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=limit)
        skill_ids = [item.skill.id for item in suggestions]
        self.project_memory.add_skill_suggestion(task, skill_ids)
        safe_task = self.project_memory.sanitize_text(task)
        return {
            "task": safe_task,
            "suggested_skills": [item.to_dict() for item in suggestions],
        }

    def skills_context(self, task: str, *, limit: int = 5, max_memory_chars: int = 2000) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=limit)
        safe_task = self.project_memory.sanitize_text(task)
        skill_contexts = [item.skill.get_context(safe_task, project_summary) for item in suggestions]
        git_data: dict = {}
        if any(item.skill.id == "git-review" for item in suggestions):
            git_receipt = self.git_summary()
            git_data = git_receipt.data.get("git", {}) if git_receipt.status == CodeActionStatus.OK else {"status": git_receipt.status.value, "message": git_receipt.message}
        skill_ids = [item.skill.id for item in suggestions]
        self.project_memory.add_skill_context(task, skill_ids)
        return {
            "task": safe_task,
            "memory_summary": self.project_memory.get_agent_context_summary(max_chars=max_memory_chars),
            "git_summary": self.project_memory.sanitize_value(git_data),
            "suggested_skills": [item.to_dict() for item in suggestions],
            "skill_contexts": self.project_memory.sanitize_value(skill_contexts),
        }

    def repos_index(self, library_root: str, *, max_repos: int | None = None) -> dict:
        result = self.project_memory.sanitize_value(self.repo_library.index(library_root, max_repos=max_repos))
        self.project_memory.add_repo_library_index(str(result.get("library_root", "")), int(result.get("repo_count", 0)), int(result.get("snippet_count", 0)))
        self._local_search_rebuild_silent("repo_library_index")
        return result

    def repos_list(self, *, limit: int = 100) -> dict:
        return self.project_memory.sanitize_value(self.repo_library.list(limit=limit))

    def repos_stats(self) -> dict:
        return self.project_memory.sanitize_value(self.repo_library.stats())

    def repos_show(self, repo_id: str) -> dict:
        return self.project_memory.sanitize_value(self.repo_library.show(repo_id))

    def repos_search(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None) -> dict:
        safe_query = self.project_memory.sanitize_text(query)
        result = self.project_memory.sanitize_value(self.repo_library.search(safe_query, skill_ids=skill_ids, limit=limit))
        self.project_memory.add_repo_library_search(query, skill_ids or [], [item.get("repo_id", "") for item in result.get("results", [])])
        return result

    def repos_search_task(self, task: str, *, limit: int = 10) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=5)
        skill_ids = [item.skill.id for item in suggestions]
        query = self._repo_task_query(task, skill_ids)
        result = self.repos_search(query, limit=limit, skill_ids=skill_ids)
        result["task"] = self.project_memory.sanitize_text(task)
        result["suggested_skills"] = [item.to_dict() for item in suggestions]
        return result

    def learn_search_github(self, query: str, *, max_results: int = 20, language: str | None = None, topic: str | None = None) -> dict:
        result = self.project_memory.sanitize_value(self.repo_learning.search_github(query, max_results=max_results, language=language, topic=topic))
        self.project_memory.add_repo_learning_event("github_search", count=int(result.get("result_count", 0)), warning=str(result.get("message", "")))
        return result

    def learn_shortlist(self, task: str, *, max_results: int = 10, language: str | None = None, topic: str | None = None) -> dict:
        safe_task = str(self.project_memory.sanitize_value(task))
        result = self.project_memory.sanitize_value(self.repo_learning.shortlist(safe_task, max_results=max_results, language=language, topic=topic))
        self.project_memory.add_repo_learning_event("github_shortlist", count=len(result.get("candidates", [])) if isinstance(result, dict) else 0)
        return result

    def learn_clone(self, repo_id: str, *, library_root: str, confirm: bool = False, overwrite: bool = False) -> dict:
        result = self.project_memory.sanitize_value(self.repo_learning.clone(repo_id, library_root=library_root, confirm=confirm, overwrite=overwrite))
        self.project_memory.add_repo_learning_event("github_clone", repo_ids=[repo_id], count=1 if result.get("status") == "ok" else 0, warning=str(result.get("message", "")))
        return result

    def learn_clone_and_index(self, repo_id: str, *, library_root: str, confirm: bool = False, overwrite: bool = False) -> dict:
        result = self.project_memory.sanitize_value(self.repo_learning.clone_and_index(repo_id, library_root=library_root, confirm=confirm, overwrite=overwrite))
        extracted = result.get("extracted") if isinstance(result.get("extracted"), dict) else {}
        self.project_memory.add_repo_learning_event("github_clone_and_index", repo_ids=[repo_id], count=int(extracted.get("entry_count", 0) or 0), warning=str(result.get("clone", {}).get("message", "") if isinstance(result.get("clone"), dict) else ""))
        if isinstance(result.get("clone"), dict) and result["clone"].get("status") == "ok":
            self._local_search_rebuild_silent("github_clone_and_index")
        return result

    def learn_extract(self) -> dict:
        result = self.project_memory.sanitize_value(self.repo_learning.extract())
        self.project_memory.add_repo_learning_event("learning_extract", count=int(result.get("entry_count", 0)))
        self._local_search_rebuild_silent("learning_extract")
        return result

    def learn_list(self, *, limit: int = 100) -> dict:
        return self.project_memory.sanitize_value(self.repo_learning.list(limit=limit))

    def learn_search(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None) -> dict:
        safe_query = self.project_memory.sanitize_text(query)
        result = self.project_memory.sanitize_value(self.repo_learning.search(safe_query, skill_ids=skill_ids, limit=limit))
        self.project_memory.add_repo_learning_event("learning_search", skill_ids=skill_ids or [], count=int(result.get("result_count", 0)))
        return result

    def learn_for_task(self, task: str, *, limit: int = 8) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=5)
        skill_ids = [item.skill.id for item in suggestions]
        safe_task = self.project_memory.sanitize_text(task)
        result = self.project_memory.sanitize_value(self.repo_learning.for_task(safe_task, skill_ids=skill_ids, limit=limit))
        result["suggested_skills"] = [item.to_dict() for item in suggestions]
        self.project_memory.add_repo_learning_event("learning_for_task", skill_ids=skill_ids, count=int(result.get("result_count", 0)))
        return result

    def learn_stats(self) -> dict:
        return self.project_memory.sanitize_value(self.repo_learning.stats())

    def learn_summary(self, *, limit: int = 8) -> dict:
        return self.project_memory.sanitize_value(self.repo_learning.summary(limit=limit))

    def local_search_rebuild(self) -> dict:
        result = self.project_memory.sanitize_value(self.local_search.rebuild())
        self.project_memory.add_local_search_event("rebuild", count=int(result.get("document_count", 0)), warning=str(result.get("warnings", "")))
        return result

    def local_search_query(self, query: str, *, limit: int = 10, skill_ids: list[str] | None = None) -> dict:
        safe_query = self.project_memory.sanitize_text(query)
        result = self.project_memory.sanitize_value(self.local_search.query(safe_query, limit=limit, skill_ids=skill_ids))
        self.project_memory.add_local_search_event("query", query=query, skill_ids=skill_ids or [], count=int(result.get("result_count", 0)), warning=str(result.get("warnings", "")))
        return result

    def local_search_task(self, task: str, *, limit: int = 10) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=5)
        skill_ids = [item.skill.id for item in suggestions]
        safe_task = self.project_memory.sanitize_text(task)
        result = self.project_memory.sanitize_value(self.local_search.task(safe_task, skill_ids=skill_ids, limit=limit))
        result["suggested_skills"] = [item.to_dict() for item in suggestions]
        self.project_memory.add_local_search_event("task", query=task, skill_ids=skill_ids, count=int(result.get("result_count", 0)), warning=str(result.get("warnings", "")))
        return result

    def local_search_context_for_task(self, task: str, *, max_results: int = 10, max_chars: int = 4000) -> dict:
        project_summary = self._skill_project_summary()
        suggestions = self.skill_router.suggest(task, project_summary, limit=5)
        skill_ids = [item.skill.id for item in suggestions]
        result = self.project_memory.sanitize_value(self.local_search.context_for_task(task, skill_ids=skill_ids, max_results=max_results, max_chars=max_chars))
        result["suggested_skills"] = [item.to_dict() for item in suggestions]
        self.project_memory.add_local_search_event("context_for_task", query=task, skill_ids=skill_ids, count=int(result.get("result_count", 0)), warning=str(result.get("warnings", "")))
        return result

    def local_search_stats(self) -> dict:
        return self.project_memory.sanitize_value(self.local_search.stats())

    def local_search_clear(self, *, confirm: bool = False) -> dict:
        result = self.project_memory.sanitize_value(self.local_search.clear(confirm=confirm))
        self.project_memory.add_local_search_event("clear", count=len(result.get("removed", [])) if isinstance(result.get("removed"), list) else 0, warning=str(result.get("message", "")))
        return result

    def agent_context(self, task: str) -> dict:
        return self.project_memory.sanitize_value(self.agent.context(task))

    def agent_plan(self, task: str, *, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> dict:
        return self.project_memory.sanitize_value(self.agent.plan(task, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited))

    def agent_run(self, task: str, *, mode: AgentRunMode | str = AgentRunMode.DRY_RUN, confirm: bool = False, pin: str | None = None, generate_patch: bool = False, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> dict:
        selected = AgentRunMode(mode)
        return self.project_memory.sanitize_value(self.agent.run(task, mode=selected, confirm=confirm, pin=pin, generate_patch=generate_patch, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited))

    def agent_run_patch(self, task: str, patch_id: str, *, mode: AgentRunMode | str = AgentRunMode.DRY_RUN, confirm: bool = False, pin: str | None = None, max_steps: int = 12, max_commands: int = 3, max_files_edited: int = 3) -> dict:
        selected = AgentRunMode(mode)
        return self.project_memory.sanitize_value(self.agent.run(task, mode=selected, confirm=confirm, pin=pin, patch_id=patch_id, max_steps=max_steps, max_commands=max_commands, max_files_edited=max_files_edited))

    def agent_verify(self) -> dict:
        return self.project_memory.sanitize_value(self.agent.verify())

    def patch_propose_replace(self, file: str, old: str, new: str, *, task: str = "replace text") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_replace(file, old, new, task=task), "propose_replace")

    def patch_propose_insert_after(self, file: str, anchor: str, text: str, *, task: str = "insert text after anchor") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_insert_after(file, anchor, text, task=task), "propose_insert_after")

    def patch_propose_insert_before(self, file: str, anchor: str, text: str, *, task: str = "insert text before anchor") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_insert_before(file, anchor, text, task=task), "propose_insert_before")

    def patch_propose_append(self, file: str, text: str, *, task: str = "append text") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_append(file, text, task=task), "propose_append")

    def patch_propose_create_file(self, file: str, content: str, *, task: str = "create file") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_create_file(file, content, task=task), "propose_create_file")

    def patch_propose_unified_diff(self, diff_text: str, *, task: str = "unified diff") -> dict:
        return self._record_patch_proposal(lambda: self.patch_builder.propose_unified_diff(diff_text, task=task), "propose_unified_diff")

    def patch_list(self, *, limit: int = 100) -> dict:
        return self.project_memory.sanitize_value(self.patch_store.list(limit=limit))

    def patch_show(self, patch_id: str) -> dict:
        return self.project_memory.sanitize_value(self.patch_store.show(patch_id))

    def patch_apply(self, patch_id: str, *, confirm: bool = False, pin: str | None = None, checkpoint: bool = False) -> dict:
        result = self.project_memory.sanitize_value(self.patch_applier.apply(patch_id, confirm=confirm, pin=pin, checkpoint=checkpoint))
        self.project_memory.add_patch_event("apply", patch_id=patch_id, files=result.get("touched_files", []), summary=result.get("message", ""), status=result.get("status", ""), commands=result.get("commands", []), warning="; ".join(result.get("errors", [])[:3]) if isinstance(result.get("errors"), list) else "")
        return result

    def patch_reject(self, patch_id: str) -> dict:
        result = self.project_memory.sanitize_value(self.patch_store.update_status(patch_id, "rejected"))
        self.project_memory.add_patch_event("reject", patch_id=patch_id, files=result.get("target_files", []), summary=result.get("summary", ""), status="rejected")
        return result

    def patch_stats(self) -> dict:
        return self.project_memory.sanitize_value(self.patch_store.stats())

    def change_targets(self, task: str, *, max_targets: int = 3) -> dict:
        return self.change_generator.targets(task, max_targets=max_targets)

    def change_plan(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False) -> dict:
        return self.change_generator.plan(task, max_targets=max_targets, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online)

    def change_propose(self, task: str, *, max_targets: int = 3, llm_assisted: bool = False, llm_mode: str | None = None, allow_online: bool = False) -> dict:
        return self.change_generator.propose(task, max_targets=max_targets, llm_assisted=llm_assisted, llm_mode=llm_mode, allow_online=allow_online)

    def llm_status(self) -> dict:
        return self.project_memory.sanitize_value(
            {
                "provider": self.llm_provider.provider_name,
                "model": self.llm_provider.model_name,
                "available": self.llm_provider.is_available(),
                "mode": self.llm_config.mode,
                "local_provider": self.llm_router.local_provider.provider_name,
                "local_available": self.llm_router.local_provider.is_available(),
                "online_provider": self.llm_router.online_provider.provider_name,
                "online_available": self.llm_router.online_provider.is_available(),
                "config_warning": self.llm_config.warning,
            }
        )

    def llm_config_view(self) -> dict:
        return self.project_memory.sanitize_value(self.llm_config.safe_dict())

    def llm_mode_view(self) -> dict:
        return self.project_memory.sanitize_value({"mode": self.llm_config.mode, "prefer_local": self.llm_config.prefer_local, "allow_online_for_code": self.llm_config.allow_online_for_code})

    def llm_route(self, task: str, *, llm_mode: str | None = None, allow_online: bool = False) -> dict:
        targets, _, _ = self.change_generator._resolver.resolve(task)
        prompt = self.project_memory.sanitize_text(task)
        _, decision = self.llm_router.route(task, targets, mode=llm_mode, allow_online_override=allow_online, prompt=prompt)
        return self.project_memory.sanitize_value(decision.__dict__)

    def git_status(self) -> CodeAgentReceipt:
        return self._git_read("git status --short --branch", lambda auth: self.git.status(auth), "git status read")

    def git_current_branch(self) -> CodeAgentReceipt:
        return self._git_read("git branch --show-current", lambda auth: {"branch": self.git.current_branch(auth)}, "git branch read")

    def git_diff(self, path: str | None = None) -> CodeAgentReceipt:
        command = "git diff"
        if path:
            try:
                target = normalize_project_path(self.project_root, path)
                path = str(target.relative_to(self.project_root))
                command = f"git diff -- {path}"
            except Exception as exc:  # noqa: BLE001
                started = utcnow()
                mode = self.current_mode()
                risk = RiskAssessment(level=RiskLevel.CRITICAL, reason="path is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])
                receipt = CodeAgentReceipt(
                    action=CodeActionKind.GIT_OPERATION,
                    status=CodeActionStatus.BLOCKED,
                    risk=risk,
                    mode=mode,
                    message=str(exc),
                    target=str(path),
                    tool="git_manager",
                    confirmation_required=True,
                    pin_required=True,
                    blocked_reason=str(exc),
                    started_at=started,
                    finished_at=utcnow(),
                )
                self.action_log.append(receipt)
                return receipt
        return self._git_read(command, lambda auth: self.git.diff(auth, path=path), "git diff read")

    def git_diff_stat(self) -> CodeAgentReceipt:
        return self._git_read("git diff --stat", lambda auth: self.git.diff_stat(auth), "git diff stat read")

    def git_changed_files(self) -> CodeAgentReceipt:
        return self._git_read("git status --short", lambda auth: {"changed_files": self.git.changed_files(auth)}, "git changed files read")

    def git_summary(self) -> CodeAgentReceipt:
        return self._git_read("git status --short", lambda auth: self.git.summary(auth), "git summary read")

    def git_checkpoint(self, *, confirm: bool = False, pin: str | None = None, message: str | None = None) -> CodeAgentReceipt:
        checkpoint_message = self.git.sanitize_checkpoint_message(message or "jarvis checkpoint")
        return self._git_write(f"git stash push -u -m {checkpoint_message}", lambda auth: self.git.create_checkpoint(auth, message=checkpoint_message), "git checkpoint created", confirm=confirm, pin=pin)

    def git_create_branch(self, name: str, *, confirm: bool = False, pin: str | None = None) -> CodeAgentReceipt:
        branch = self.git.sanitize_branch_name(name)
        exists = self._git_read(f"git branch --list {branch}", lambda auth: {"branch": branch, "exists": self.git.branch_exists(branch, auth)}, "git branch existence read")
        if exists.status != CodeActionStatus.OK or not exists.data.get("git", {}).get("is_repo", False):
            return exists
        if exists.data.get("git", {}).get("exists"):
            started = utcnow()
            mode = self.current_mode()
            risk = RiskAssessment(level=RiskLevel.MINOR_CHANGE, reason="branch already exists", tags=["git_branch"])
            receipt = CodeAgentReceipt(
                action=CodeActionKind.GIT_OPERATION,
                status=CodeActionStatus.FAILED,
                risk=risk,
                mode=mode,
                message=f"branch already exists: {branch}",
                target=str(self.project_root),
                tool="git_manager",
                commands=[f"git branch --list {branch}"],
                data={"git": {"is_repo": True, "branch": branch, "exists": True}},
                started_at=started,
                finished_at=utcnow(),
            )
            self.action_log.append(receipt)
            return receipt
        return self._git_write(f"git switch -c {branch}", lambda auth: self.git.create_branch(branch, auth), "git branch created", confirm=confirm, pin=pin)

    def git_revert_file(self, path: str, *, confirm: bool = False, pin: str | None = None) -> CodeAgentReceipt:
        try:
            target = normalize_project_path(self.project_root, path)
            rel_path = str(target.relative_to(self.project_root))
        except Exception as exc:  # noqa: BLE001
            started = utcnow()
            mode = self.current_mode()
            risk = RiskAssessment(level=RiskLevel.CRITICAL, reason="path is outside project root", requires_confirmation=True, requires_pin=True, tags=["outside_project"])
            receipt = CodeAgentReceipt(
                action=CodeActionKind.GIT_OPERATION,
                status=CodeActionStatus.BLOCKED,
                risk=risk,
                mode=mode,
                message=str(exc),
                target=path,
                tool="git_manager",
                confirmation_required=True,
                pin_required=True,
                blocked_reason=str(exc),
                started_at=started,
                finished_at=utcnow(),
            )
            self.action_log.append(receipt)
            return receipt
        return self._git_write(f"git checkout -- {rel_path}", lambda auth: self.git.revert_file(rel_path, auth), "git file revert completed", confirm=confirm, pin=pin)

    def _git_read(self, command: str, operation, message: str) -> CodeAgentReceipt:
        return self._git_operation(command, operation, message, confirm=False, pin=None)

    def _git_write(self, command: str, operation, message: str, *, confirm: bool, pin: str | None) -> CodeAgentReceipt:
        return self._git_operation(command, operation, message, confirm=confirm, pin=pin)

    def _git_operation(self, command: str, operation, message: str, *, confirm: bool, pin: str | None) -> CodeAgentReceipt:
        started = utcnow()
        mode = self.current_mode()
        risk = self.risk_classifier.assess_command(command, self.project_root)
        blocked = self._check_permission(risk, action=CodeActionKind.GIT_OPERATION, target=command, started=started, confirm=confirm, pin=pin, mode=mode, command=command, tool="git_manager")
        if blocked is not None:
            return blocked
        try:
            authorization = self._authorization_context(action=CodeActionKind.GIT_OPERATION, target=command, risk=risk, mode=mode, confirm=confirm)
            if not self.git.is_repo():
                data = {"is_repo": False, "message": "project root is not a Git repository"}
            else:
                data = operation(authorization)
                data["is_repo"] = True
                if "branch" in data:
                    self.project_memory.add_note(f"Git branch: {data['branch']}")
                if "changed_files" in data:
                    self.project_memory.add_note(f"Git changed files: {len(data['changed_files'])}")
                if "message" in data and "checkpoint" in message:
                    self.project_memory.add_decision("Git checkpoint created", str(data["message"]), [])
            receipt = CodeAgentReceipt(
                action=CodeActionKind.GIT_OPERATION,
                status=CodeActionStatus.OK,
                risk=risk,
                mode=mode,
                message=message if data.get("is_repo", False) else "not a git repository",
                target=str(self.project_root),
                tool="git_manager",
                commands=[command],
                data={"git": data},
                pin_verified=True if risk.requires_pin else None,
                started_at=started,
                finished_at=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            self.project_memory.add_failed_command(command, str(exc), int(risk.level))
            receipt = self._failure(CodeActionKind.GIT_OPERATION, risk, command, started, exc, mode=mode, tool="git_manager")
        self.action_log.append(receipt)
        return receipt

    def _check_permission(
        self,
        risk: RiskAssessment,
        *,
        action: CodeActionKind,
        target: str,
        started,
        mode: OperationMode,
        confirm: bool = False,
        pin: str | None = None,
        path: Path | None = None,
        command: str | None = None,
        tool: str | None = None,
    ) -> CodeAgentReceipt | None:
        decision = self.permission_gate.evaluate(action=action, mode=mode, risk=risk, path=path, command=command, confirm=confirm, pin=pin)
        if decision.allowed:
            return None
        status = self._status_for_decision(decision)
        receipt = CodeAgentReceipt(
            action=action,
            status=status,
            risk=risk,
            mode=mode,
            message=decision.reason,
            target=target,
            tool=tool,
            data={"confirmation": decision.confirmation_prompt} if decision.confirmation_prompt else {},
            confirmation_required=decision.requires_confirmation,
            pin_required=decision.requires_pin,
            pin_verified=decision.pin_verified,
            blocked_reason=decision.reason if status == CodeActionStatus.BLOCKED else None,
            started_at=started,
            finished_at=utcnow(),
        )
        self.action_log.append(receipt)
        return receipt

    def _local_search_rebuild_silent(self, action: str) -> None:
        try:
            result = self.project_memory.sanitize_value(self.local_search.rebuild())
            self.project_memory.add_local_search_event(action, count=int(result.get("document_count", 0)), warning=str(result.get("warnings", "")))
        except Exception as exc:  # noqa: BLE001
            self.project_memory.add_local_search_event(action, count=0, warning=str(exc))

    def _record_patch_proposal(self, callback_result, action: str) -> dict:
        try:
            result = self.project_memory.sanitize_value(callback_result())
        except Exception as exc:  # noqa: BLE001
            result = {"status": "blocked", "message": str(exc), "target_files": []}
        self.project_memory.add_patch_event(
            action,
            patch_id=str(result.get("id", "")),
            files=result.get("target_files", []) if isinstance(result.get("target_files"), list) else [],
            summary=str(result.get("summary", result.get("message", ""))),
            status=str(result.get("status", "")),
            skills=result.get("skills", []) if isinstance(result.get("skills"), list) else [],
            warning=str(result.get("warnings", result.get("message", ""))),
        )
        return result

    def _skill_project_summary(self) -> dict:
        memory = self.project_memory.load()
        project_structure = memory.get("project_structure") if isinstance(memory.get("project_structure"), dict) else {}
        return {
            "important_files": list(memory.get("important_files", [])),
            "key_files": list(project_structure.get("key_files", [])) if isinstance(project_structure, dict) else [],
            "languages": project_structure.get("languages", {}) if isinstance(project_structure, dict) else {},
            "scripts": project_structure.get("scripts", {}) if isinstance(project_structure, dict) else {},
            "internal_modules": project_structure.get("internal_modules", []) if isinstance(project_structure, dict) else [],
        }

    @staticmethod
    def _repo_task_query(task: str, skill_ids: list[str]) -> str:
        terms = [task]
        expansions = {
            "python": "python pytest import traceback",
            "testing": "tests pytest fixture assert regression",
            "debugging": "traceback error exception fix root cause",
            "frontend-react": "react typescript tsx component props state",
            "security-audit": "permissions auth security path traversal command injection secret",
            "cli": "cli typer argparse click command",
            "git-review": "git diff status checkpoint branch",
            "docs": "readme documentation examples",
        }
        terms.extend(expansions[skill_id] for skill_id in skill_ids if skill_id in expansions)
        return " ".join(terms)

    @staticmethod
    def _authorization_context(
        *,
        action: CodeActionKind,
        target: str,
        risk: RiskAssessment,
        mode: OperationMode,
        confirm: bool,
    ) -> AuthorizationContext:
        return AuthorizationContext(
            action=action,
            target=target,
            risk=risk,
            mode=mode,
            allowed=True,
            confirmation_confirmed=confirm if risk.requires_confirmation else False,
            pin_verified=True if risk.requires_pin else None,
        )

    @staticmethod
    def _status_for_decision(decision: PermissionResult) -> CodeActionStatus:
        if decision.decision == PermissionDecision.REQUIRE_CONFIRMATION:
            return CodeActionStatus.CONFIRMATION_REQUIRED
        if decision.decision == PermissionDecision.REQUIRE_PIN:
            return CodeActionStatus.CONFIRMATION_REQUIRED
        return CodeActionStatus.BLOCKED

    @staticmethod
    def _failure(action: CodeActionKind, risk: RiskAssessment, target: str, started, exc: Exception, *, mode: OperationMode, tool: str | None = None) -> CodeAgentReceipt:
        return CodeAgentReceipt(
            action=action,
            status=CodeActionStatus.FAILED,
            risk=risk,
            mode=mode,
            message=str(exc),
            target=target,
            tool=tool,
            errors=[str(exc)],
            started_at=started,
            finished_at=utcnow(),
        )
