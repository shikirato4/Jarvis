from __future__ import annotations

from math import sqrt
from typing import Any, Protocol

from .base import RetrievedChunk
from .repository import SemanticMemoryRepository


class VectorIndex(Protocol):
    def upsert(self, collection_name: str, *, document_id: str, chunk_ids: list[str]) -> None: ...

    def search(
        self,
        *,
        query_vector: list[float],
        collection_name: str | None,
        top_k: int,
        min_score: float,
        source_types: tuple[str, ...] = (),
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def rebuild(self, collection_name: str | None = None) -> int: ...


class RepositoryVectorIndex:
    def __init__(self, repository: SemanticMemoryRepository) -> None:
        self._repository = repository

    def upsert(self, collection_name: str, *, document_id: str, chunk_ids: list[str]) -> None:
        return

    def search(
        self,
        *,
        query_vector: list[float],
        collection_name: str | None,
        top_k: int,
        min_score: float,
        source_types: tuple[str, ...] = (),
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        metadata_filters = metadata_filters or {}
        candidates = self._repository.list_chunks(
            collection_name=collection_name,
            source_types=source_types,
            with_embeddings_only=True,
        )
        scored: list[RetrievedChunk] = []
        for chunk in candidates:
            if metadata_filters and any(chunk.metadata.get(key) != value for key, value in metadata_filters.items()):
                continue
            score = _cosine_similarity(query_vector, chunk.embedding_vector)
            if score < min_score:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    collection_name=chunk.collection_name,
                    document_title=chunk.metadata.get("document_title"),
                    source_path=chunk.provenance.source_path,
                    source_type=chunk.source_type.value,
                    content=chunk.content,
                    score=score,
                    rank=0,
                    metadata=chunk.metadata,
                    provenance=chunk.provenance.model_dump(mode="json"),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        results = scored[:top_k]
        for index, chunk in enumerate(results, start=1):
            chunk.rank = index
        return results

    def rebuild(self, collection_name: str | None = None) -> int:
        return len(
            self._repository.list_chunks(
                collection_name=collection_name,
                with_embeddings_only=True,
            )
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
