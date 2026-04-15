from __future__ import annotations

from uuid import uuid4

from jarvis.core.errors import EmbeddingProviderError
from jarvis.memory_semantic.base import EmbeddingRequest
from jarvis.memory_semantic.documents import DocumentIngestionRequest

from .models import IndexedChunk, IndexedDocument, IndexSource


class IndexEmbeddingCoordinator:
    def __init__(self, embedding_service, semantic_memory_service) -> None:
        self._embedding_service = embedding_service
        self._semantic_memory = semantic_memory_service

    def embed(self, document: IndexedDocument, chunks: list[IndexedChunk], source: IndexSource) -> tuple[IndexedDocument, list[IndexedChunk]]:
        if not chunks:
            return document, chunks
        try:
            response = self._embedding_service.embed(
                EmbeddingRequest(
                    texts=tuple(chunk.text for chunk in chunks),
                    task_type="ingestion",
                    correlation_id=f"index-{uuid4()}",
                    metadata={"source_id": source.source_id, "document_uri": document.canonical_uri},
                )
            )
            for chunk, vector in zip(chunks, response.vectors, strict=False):
                chunk.embedding_vector = list(vector.values)
                chunk.embedding_model = response.model_name
                chunk.embedding_provider = response.provider_name
                chunk.embedding_dim = vector.dimensions
        except Exception as exc:  # noqa: BLE001
            if not source.embedding_policy.get("allow_lexical_only", True):
                raise EmbeddingProviderError(str(exc)) from exc
        if source.embedding_policy.get("semantic_projection", True):
            try:
                semantic_doc = self._semantic_memory.ingest_document(
                    DocumentIngestionRequest(
                        collection_name=source.collection_name or f"index_{source.source_id}",
                        source_type=self._semantic_source_type(str(document.document_type.value)),
                        path=document.path,
                        content=None if document.path else document.content,
                        title=document.title,
                        metadata={
                            **document.metadata,
                            "index_source_id": source.source_id,
                            "canonical_uri": document.canonical_uri,
                        },
                        persist_memory=False,
                    )
                )
                document.semantic_document_id = semantic_doc.id
                document.semantic_collection_name = semantic_doc.collection_name
            except Exception as exc:  # noqa: BLE001
                if not source.embedding_policy.get("allow_lexical_only", True):
                    raise EmbeddingProviderError(str(exc)) from exc
        return document, chunks

    @staticmethod
    def _semantic_source_type(document_type: str) -> str:
        mapping = {
            "markdown": "markdown",
            "json": "json",
            "research_report": "research_note",
            "writing_context": "draft",
            "unity_asset": "text",
            "code": "text",
            "pdf": "book",
        }
        return mapping.get(document_type, "text")
