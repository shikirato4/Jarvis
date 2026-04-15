from __future__ import annotations

from typing import Iterable

from .editor_session import UnityEditorSession


class UnityEditorSessionRegistry:
    def __init__(self) -> None:
        self._by_project: dict[str, UnityEditorSession] = {}

    def save(self, session: UnityEditorSession) -> UnityEditorSession:
        self._by_project[session.project_root] = session
        return session

    def get(self, project_root: str) -> UnityEditorSession | None:
        return self._by_project.get(project_root)

    def remove(self, project_root: str) -> UnityEditorSession | None:
        return self._by_project.pop(project_root, None)

    def list_sessions(self) -> list[UnityEditorSession]:
        return list(self._by_project.values())

    def list_by_status(self, status: str) -> list[UnityEditorSession]:
        return [item for item in self._by_project.values() if item.status.value == status]

    def update_errors(self, project_root: str, error: str) -> UnityEditorSession | None:
        session = self.get(project_root)
        if session is None:
            return None
        updated = session.mark_degraded(reason=error)
        self.save(updated)
        return updated
