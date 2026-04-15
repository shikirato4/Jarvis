from __future__ import annotations

from typing import Protocol

from .models import DiscoveredSourceItem, IndexedChunk, IndexedDocument, IndexSource


class IndexingDocumentLoader(Protocol):
    def load(self, item: DiscoveredSourceItem, source: IndexSource) -> IndexedDocument: ...


class IndexingChunker(Protocol):
    def chunk(self, document: IndexedDocument, source: IndexSource) -> list[IndexedChunk]: ...


class IndexingEmbedder(Protocol):
    def embed(
        self,
        document: IndexedDocument,
        chunks: list[IndexedChunk],
        source: IndexSource,
    ) -> tuple[IndexedDocument, list[IndexedChunk]]: ...
