from __future__ import annotations

import logging
from collections import OrderedDict

from .base import EmbeddingRequest, RetrievedChunk, RetrievedContext, SemanticSearchQuery
from .embeddings import EmbeddingService
from .index import VectorIndex
from .ranking import RetrievalReranker
from .repository import SemanticMemoryRepository


class RetrievalPipeline:
    def __init__(
        self,
        repository: SemanticMemoryRepository,
        vector_index: VectorIndex,
        reranker: RetrievalReranker,
        embedding_service: EmbeddingService,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repository = repository
        self._vector_index = vector_index
        self._reranker = reranker
        self._embedding_service = embedding_service
        self._logger = logger or logging.getLogger("jarvis.semantic.retrieval")

    def retrieve(self, query: SemanticSearchQuery) -> RetrievedContext:
        top_k = query.top_k or 5
        min_score = query.min_score if query.min_score is not None else 0.0
        fallback_applied = False
        degraded = False
        strategy = "semantic"
        chunks: list[RetrievedChunk]
        try:
            embedding = self._embedding_service.embed(
                EmbeddingRequest(
                    texts=(query.query,),
                    logical_model=None,
                    task_type="retrieval",
                    correlation_id=query.correlation_id,
                    metadata={"collection_name": query.collection_name},
                )
            )
            query_vector = embedding.vectors[0].values if embedding.vectors else []
            chunks = self._vector_index.search(
                query_vector=query_vector,
                collection_name=query.collection_name,
                top_k=top_k,
                min_score=min_score,
                source_types=query.source_types,
                metadata_filters=query.metadata_filters,
            )
        except Exception:
            chunks = []
            fallback_applied = True
            degraded = True
            strategy = "lexical_fallback"
            self._logger.warning("semantic_retrieval_degraded", extra={"query": query.query, "collection": query.collection_name})

        if not chunks:
            lexical = self._repository.lexical_search(
                query=query.query,
                collection_name=query.collection_name,
                source_types=query.source_types,
                limit=top_k,
            )
            if lexical:
                fallback_applied = True
                degraded = True
                strategy = "lexical_fallback"
                chunks = [
                    RetrievedChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        collection_name=chunk.collection_name,
                        document_title=chunk.metadata.get("document_title"),
                        source_path=chunk.provenance.source_path,
                        source_type=chunk.source_type.value,
                        content=chunk.content,
                        score=1.0 / (index + 1),
                        rank=index + 1,
                        metadata=chunk.metadata,
                        provenance=chunk.provenance.model_dump(mode="json"),
                    )
                    for index, chunk in enumerate(lexical)
                ]

        chunks = self._deduplicate(chunks)
        if chunks:
            chunks = self._reranker.rerank(query.query, chunks)
        summary = self._build_summary(chunks) if query.include_summary else None
        return RetrievedContext(
            query=query.query,
            strategy=strategy,
            chunks=chunks[:top_k],
            sources=list(OrderedDict.fromkeys(chunk.source_path for chunk in chunks if chunk.source_path)),
            summary=summary,
            total_chunks=len(chunks),
            degraded=degraded,
            fallback_applied=fallback_applied,
            metadata={"collection_name": query.collection_name, "filters": query.metadata_filters},
        )

    @staticmethod
    def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        seen: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            deduped.append(chunk)
        for index, chunk in enumerate(deduped, start=1):
            chunk.rank = index
        return deduped

    @staticmethod
    def _build_summary(chunks: list[RetrievedChunk]) -> str | None:
        if not chunks:
            return None
        lines = [f"[{chunk.rank}] {chunk.content[:220].strip()}" for chunk in chunks[:5]]
        return "\n".join(lines)
