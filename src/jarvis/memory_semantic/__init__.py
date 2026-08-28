from .base import RetrievedContext, SemanticSearchQuery, SemanticSearchResult
from .documents import DocumentIngestionRequest

__all__ = [
    "DocumentIngestionRequest",
    "RetrievedContext",
    "SemanticMemoryService",
    "SemanticSearchQuery",
    "SemanticSearchResult",
]


def __getattr__(name: str):
    if name == "SemanticMemoryService":
        from .service import SemanticMemoryService

        return SemanticMemoryService
    raise AttributeError(name)
