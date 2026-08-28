"""Task routing layer for top-level runtime requests."""

from .models import TaskRequest, TaskResponse

__all__ = ["TaskRequest", "TaskResponse", "TaskRouter"]


def __getattr__(name: str):
    if name == "TaskRouter":
        from .task_router import TaskRouter

        return TaskRouter
    raise AttributeError(name)
