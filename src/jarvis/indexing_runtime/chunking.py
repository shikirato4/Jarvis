from __future__ import annotations

import hashlib
from uuid import uuid4

from jarvis.ingestion.chunking import TextChunker

from .models import IndexedChunk, IndexedDocument, IndexSource


class IntelligentIndexChunker:
    def chunk(self, document: IndexedDocument, source: IndexSource) -> list[IndexedChunk]:
        chunk_size = int(source.chunking_policy.get("chunk_size", 900))
        chunk_overlap = int(source.chunking_policy.get("chunk_overlap", 120))
        max_chunks = int(source.chunking_policy.get("max_chunks", 200))
        raw_chunks = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(document.content)
        chunks: list[IndexedChunk] = []
        cursor = 0
        for raw in raw_chunks[:max_chunks]:
            start_offset = document.content.find(raw.content, cursor)
            if start_offset < 0:
                start_offset = cursor
            end_offset = start_offset + len(raw.content)
            cursor = end_offset
            chunks.append(
                IndexedChunk(
                    chunk_id=str(uuid4()),
                    document_id=document.document_id,
                    snapshot_id=document.snapshot_id,
                    chunk_index=raw.index,
                    text=raw.content,
                    char_count=raw.char_count,
                    token_estimate=raw.token_estimate,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    fingerprint=hashlib.sha256(f"{document.document_id}:{raw.index}:{raw.content}".encode("utf-8")).hexdigest(),
                    lexical_terms_hash=hashlib.sha256(raw.content.casefold().encode("utf-8")).hexdigest(),
                    metadata=document.metadata.copy(),
                    provenance=document.provenance.copy(),
                )
            )
        return chunks
