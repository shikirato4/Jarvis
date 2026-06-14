from __future__ import annotations

from jarvis.code_agent_runtime.base import CodeActionKind, CodeTaskRequest
from jarvis.code_agent_runtime.executor import CodeAgentExecutor
from jarvis.code_agent_runtime.planner import CodeAgentPlanner


class CodeAgentToolRouter:
    def __init__(self, executor: CodeAgentExecutor, planner: CodeAgentPlanner | None = None) -> None:
        self._executor = executor
        self._planner = planner or CodeAgentPlanner()

    def route(self, request: CodeTaskRequest):
        action = self._planner.plan(request)
        if action == CodeActionKind.PROJECT_SCAN:
            return self._executor.scan_project()
        if action == CodeActionKind.FILE_READ:
            if not request.path:
                raise ValueError("read task requires path")
            return self._executor.read_file(request.path)
        if action == CodeActionKind.PROJECT_SEARCH:
            if not request.query:
                raise ValueError("search task requires query")
            mode = "name" if request.task.casefold() == "search-name" else "content"
            return self._executor.search_project(request.query, mode=mode)
        if action == CodeActionKind.FILE_WRITE:
            if not request.path or request.content is None:
                raise ValueError("write task requires path and content")
            return self._executor.write_file(request.path, request.content, overwrite=True, confirm=request.confirm, pin=request.pin, dry_run=request.dry_run)
        if action == CodeActionKind.COMMAND_RUN:
            if not request.command:
                raise ValueError("run task requires command")
            return self._executor.run_command(request.command, confirm=request.confirm, pin=request.pin, dry_run=request.dry_run)
        raise ValueError(f"unsupported action: {action}")
