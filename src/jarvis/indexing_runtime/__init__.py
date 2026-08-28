from .models import IndexJob, IndexSource, IndexSourceCreateRequest, IndexStatus, IndexRunRequest

__all__ = [
    "IndexJob",
    "IndexSource",
    "IndexSourceCreateRequest",
    "IndexStatus",
    "IndexRunRequest",
    "IndexingRepository",
    "IndexingRuntimeService",
]


def __getattr__(name: str):
    if name == "IndexingRuntimeService":
        from .service import IndexingRuntimeService

        return IndexingRuntimeService
    if name == "IndexingRepository":
        from .storage import IndexingRepository

        return IndexingRepository
    raise AttributeError(name)
