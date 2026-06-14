from __future__ import annotations

from jarvis.code_agent_runtime.base import CodeActionKind, CodeTaskRequest


class CodeAgentPlanner:
    def plan(self, request: CodeTaskRequest) -> CodeActionKind:
        task = request.task.casefold().strip()
        if task in {"scan", "explore", "summary", "resumen", "explorar"}:
            return CodeActionKind.PROJECT_SCAN
        if task in {"read", "leer"}:
            return CodeActionKind.FILE_READ
        if task in {"search", "buscar", "search-content", "search-name"}:
            return CodeActionKind.PROJECT_SEARCH
        if task in {"write", "create", "edit", "escribir", "crear", "editar"}:
            return CodeActionKind.FILE_WRITE
        if task in {"run", "command", "terminal", "ejecutar"}:
            return CodeActionKind.COMMAND_RUN
        raise ValueError(f"unsupported code-agent task: {request.task}")
