from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from jarvis.config import Settings
from jarvis.core.errors import CapabilityUnavailableError
from jarvis.memory.service import MemoryService

from jarvis.ingestion.chunking import TextChunker
from jarvis.ingestion.loaders import DocumentLoader
from jarvis.ingestion.normalization import build_provenance, normalize_text_content

from .base import EmbeddingRequest, RetrievedContext, SemanticSearchQuery, SemanticSearchResult
from .documents import ChunkRecord, CollectionRecord, DocumentIngestionRequest, DocumentRecord
from .embeddings import EmbeddingService
from .index import VectorIndex
from .repository import SemanticMemoryRepository
from .retrieval import RetrievalPipeline


class SemanticMemoryService:
    def __init__(
        self,
        settings: Settings,
        repository: SemanticMemoryRepository,
        embedding_service: EmbeddingService,
        vector_index: VectorIndex,
        retrieval_pipeline: RetrievalPipeline,
        memory_service: MemoryService,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._embedding_service = embedding_service
        self._vector_index = vector_index
        self._retrieval_pipeline = retrieval_pipeline
        self._memory_service = memory_service
        self._logger = logger or logging.getLogger("jarvis.semantic")

    def create_schema(self) -> None:
        self._repository.create_schema()

    def ingest_document(self, request: DocumentIngestionRequest) -> DocumentRecord:
        collection = self._repository.upsert_collection(
            CollectionRecord(name=request.collection_name, metadata={"source_type": request.source_type.value})
        )
        loader = DocumentLoader((self._settings.resolved_workspace_root, *self._settings.resolved_research_roots))
        if request.path:
            loaded = loader.load(request.path, source_type=request.source_type, title=request.title)
            title = request.title or loaded.title
            content = loaded.content
            provenance = loaded.provenance.model_copy(update=request.provenance.model_dump(exclude_none=True, exclude={"imported_at", "extra"}))
        else:
            content = normalize_text_content(request.content or "")
            title = request.title or "Untitled semantic document"
            source_path = Path(request.path) if request.path else None
            provenance = build_provenance(source_path=source_path, content=content).model_copy(
                update=request.provenance.model_dump(exclude_none=True, exclude={"imported_at", "extra"})
            )

        document = self._repository.save_document(
            DocumentRecord(
                id=str(uuid4()),
                collection_name=collection.name,
                title=title,
                source_type=request.source_type,
                content=content,
                metadata=request.metadata,
                provenance=provenance,
            )
        )
        chunks = self._build_chunks(document, request)
        vectors = self._embedding_service.embed(
            EmbeddingRequest(
                texts=tuple(chunk.content for chunk in chunks),
                task_type="ingestion",
                correlation_id=str(uuid4()),
                metadata={"collection_name": request.collection_name, "document_id": document.id},
            )
        )
        for chunk, vector in zip(chunks, vectors.vectors, strict=False):
            chunk.embedding_vector = vector.values
            chunk.embedding_model = vectors.model_name
            chunk.embedding_provider = vectors.provider_name
        stored_chunks = self._repository.replace_chunks(document.id, chunks)
        self._vector_index.upsert(document.collection_name, document_id=document.id, chunk_ids=[chunk.id for chunk in stored_chunks])
        if request.persist_memory:
            self._memory_service.store_memory(
                kind="semantic.document",
                content=document.title,
                source="semantic_memory",
                metadata={"collection_name": document.collection_name, "document_id": document.id},
            )
        return document

    def search(self, query: SemanticSearchQuery) -> SemanticSearchResult:
        return SemanticSearchResult(query=query, context=self.retrieve_context(query))

    def retrieve_context(self, query: SemanticSearchQuery) -> RetrievedContext:
        return self._retrieval_pipeline.retrieve(query)

    def delete_document(self, document_id: str) -> None:
        self._repository.delete_document(document_id)

    def list_collections(self) -> list[dict[str, object]]:
        collections = self._repository.list_collections()
        documents = self._repository.list_documents()
        counts: dict[str, int] = {}
        for document in documents:
            counts[document.collection_name] = counts.get(document.collection_name, 0) + 1
        return [
            {
                **collection.model_dump(mode="json"),
                "document_count": counts.get(collection.name, 0),
            }
            for collection in collections
        ]

    def status(self) -> dict[str, object]:
        return {
            "providers": [entry.model_dump(mode="json") for entry in self._embedding_service.health()],
            "profiles": self._embedding_service.list_models(),
            "stats": self._repository.get_collection_stats(),
            "degradation_policy": self._settings.semantic_degradation_policy,
            "reranking": self._settings.semantic_reranking_type,
        }

    def reindex(self, collection_name: str | None = None) -> dict[str, object]:
        indexed = self._vector_index.rebuild(collection_name=collection_name)
        return {"collection_name": collection_name, "indexed_chunks": indexed}

    def _build_chunks(self, document: DocumentRecord, request: DocumentIngestionRequest) -> list[ChunkRecord]:
        chunk_size = request.chunk_size or self._settings.semantic_chunk_size
        chunk_overlap = request.chunk_overlap or self._settings.semantic_chunk_overlap
        chunk_limit = self._settings.semantic_limits_by_source_type.get(document.source_type.value, self._settings.semantic_max_chunks_per_document)
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        raw_chunks = chunker.split(document.content)
        if len(raw_chunks) > chunk_limit:
            raw_chunks = raw_chunks[:chunk_limit]
        if not raw_chunks:
            raise CapabilityUnavailableError("document produced no semantic chunks")
        return [
            ChunkRecord(
                id=str(uuid4()),
                document_id=document.id,
                collection_name=document.collection_name,
                chunk_index=item.index,
                content=item.content,
                token_estimate=item.token_estimate,
                char_count=item.char_count,
                source_type=document.source_type,
                metadata={**document.metadata, "document_title": document.title},
                provenance=document.provenance,
            )
            for item in raw_chunks
        ]
