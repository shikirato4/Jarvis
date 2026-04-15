from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String, Text, delete, desc, select
from sqlalchemy.orm import Mapped, mapped_column

from jarvis.memory.models import Base
from jarvis.memory.repository import Database

from .documents import ChunkRecord, CollectionRecord, DocumentProvenance, DocumentRecord, SourceType


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _loads(raw: str) -> dict[str, Any]:
    return json.loads(raw or "{}")


def _dump_vector(values: list[float]) -> str:
    return json.dumps(values)


def _load_vector(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(value) for value in json.loads(raw)]


class SemanticCollectionRow(Base):
    __tablename__ = "semantic_collections"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SemanticDocumentRow(Base):
    __tablename__ = "semantic_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    collection_name: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SemanticChunkRow(Base):
    __tablename__ = "semantic_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    collection_name: Mapped[str] = mapped_column(String(128), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, index=True)
    content: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    embedding_vector_json: Mapped[str] = mapped_column(Text, default="[]")
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SemanticMemoryRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    def create_schema(self) -> None:
        self._database.create_schema()

    def upsert_collection(self, collection: CollectionRecord) -> CollectionRecord:
        with self._database.session_scope() as session:
            row = session.get(SemanticCollectionRow, collection.name)
            if row is None:
                row = SemanticCollectionRow(
                    name=collection.name,
                    description=collection.description,
                    metadata_json=_dumps(collection.metadata),
                )
                session.add(row)
            else:
                row.description = collection.description
                row.metadata_json = _dumps(collection.metadata)
            session.flush()
            session.refresh(row)
            return self._to_collection(row)

    def save_document(self, document: DocumentRecord) -> DocumentRecord:
        with self._database.session_scope() as session:
            row = session.get(SemanticDocumentRow, document.id)
            if row is None:
                row = SemanticDocumentRow(
                    id=document.id,
                    collection_name=document.collection_name,
                    title=document.title,
                    source_type=document.source_type.value,
                    content=document.content,
                    metadata_json=_dumps(document.metadata),
                    provenance_json=document.provenance.model_dump_json(),
                )
                session.add(row)
            else:
                row.collection_name = document.collection_name
                row.title = document.title
                row.source_type = document.source_type.value
                row.content = document.content
                row.metadata_json = _dumps(document.metadata)
                row.provenance_json = document.provenance.model_dump_json()
            session.flush()
            session.refresh(row)
            return self._to_document(row)

    def replace_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        with self._database.session_scope() as session:
            session.execute(delete(SemanticChunkRow).where(SemanticChunkRow.document_id == document_id))
            for chunk in chunks:
                session.add(
                    SemanticChunkRow(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        collection_name=chunk.collection_name,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        token_estimate=chunk.token_estimate,
                        char_count=chunk.char_count,
                        source_type=chunk.source_type.value,
                        metadata_json=_dumps(chunk.metadata),
                        provenance_json=chunk.provenance.model_dump_json(),
                        embedding_vector_json=_dump_vector(chunk.embedding_vector),
                        embedding_model=chunk.embedding_model,
                        embedding_provider=chunk.embedding_provider,
                    )
                )
            session.flush()
            rows = list(
                session.scalars(
                    select(SemanticChunkRow)
                    .where(SemanticChunkRow.document_id == document_id)
                    .order_by(SemanticChunkRow.chunk_index.asc())
                )
            )
            return [self._to_chunk(row) for row in rows]

    def delete_document(self, document_id: str) -> None:
        with self._database.session_scope() as session:
            session.execute(delete(SemanticChunkRow).where(SemanticChunkRow.document_id == document_id))
            session.execute(delete(SemanticDocumentRow).where(SemanticDocumentRow.id == document_id))

    def list_collections(self) -> list[CollectionRecord]:
        with self._database.session_scope() as session:
            rows = list(session.scalars(select(SemanticCollectionRow).order_by(SemanticCollectionRow.name.asc())))
            return [self._to_collection(row) for row in rows]

    def list_documents(self, collection_name: str | None = None) -> list[DocumentRecord]:
        with self._database.session_scope() as session:
            statement = select(SemanticDocumentRow).order_by(desc(SemanticDocumentRow.updated_at))
            if collection_name:
                statement = statement.where(SemanticDocumentRow.collection_name == collection_name)
            return [self._to_document(row) for row in session.scalars(statement)]

    def list_chunks(
        self,
        *,
        document_id: str | None = None,
        collection_name: str | None = None,
        source_types: tuple[str, ...] = (),
        with_embeddings_only: bool = False,
    ) -> list[ChunkRecord]:
        with self._database.session_scope() as session:
            statement = select(SemanticChunkRow).order_by(SemanticChunkRow.collection_name.asc(), SemanticChunkRow.chunk_index.asc())
            if document_id:
                statement = statement.where(SemanticChunkRow.document_id == document_id)
            if collection_name:
                statement = statement.where(SemanticChunkRow.collection_name == collection_name)
            if source_types:
                statement = statement.where(SemanticChunkRow.source_type.in_(source_types))
            rows = [self._to_chunk(row) for row in session.scalars(statement)]
            if with_embeddings_only:
                return [row for row in rows if row.embedding_vector]
            return rows

    def lexical_search(
        self,
        *,
        query: str,
        collection_name: str | None,
        source_types: tuple[str, ...],
        limit: int,
    ) -> list[ChunkRecord]:
        lowered_terms = [term.casefold() for term in query.split() if term.strip()]
        candidates = self.list_chunks(collection_name=collection_name, source_types=source_types)
        scored: list[tuple[int, ChunkRecord]] = []
        for chunk in candidates:
            text = chunk.content.casefold()
            score = sum(text.count(term) for term in lowered_terms)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (item[0], item[1].chunk_index), reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def get_collection_stats(self) -> dict[str, Any]:
        return {
            "collections": len(self.list_collections()),
            "documents": len(self.list_documents()),
            "chunks": len(self.list_chunks()),
        }

    @staticmethod
    def _to_collection(row: SemanticCollectionRow) -> CollectionRecord:
        return CollectionRecord(
            name=row.name,
            description=row.description,
            metadata=_loads(row.metadata_json),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_document(row: SemanticDocumentRow) -> DocumentRecord:
        return DocumentRecord(
            id=row.id,
            collection_name=row.collection_name,
            title=row.title,
            source_type=SourceType(row.source_type),
            content=row.content,
            metadata=_loads(row.metadata_json),
            provenance=DocumentProvenance.model_validate_json(row.provenance_json or "{}"),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_chunk(row: SemanticChunkRow) -> ChunkRecord:
        return ChunkRecord(
            id=row.id,
            document_id=row.document_id,
            collection_name=row.collection_name,
            chunk_index=row.chunk_index,
            content=row.content,
            token_estimate=row.token_estimate,
            char_count=row.char_count,
            source_type=SourceType(row.source_type),
            metadata=_loads(row.metadata_json),
            provenance=DocumentProvenance.model_validate_json(row.provenance_json or "{}"),
            embedding_vector=_load_vector(row.embedding_vector_json),
            embedding_model=row.embedding_model,
            embedding_provider=row.embedding_provider,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
