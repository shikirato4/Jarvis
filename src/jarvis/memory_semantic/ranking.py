from __future__ import annotations

from typing import Protocol

from .base import RetrievedChunk


class RetrievalReranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class BasicHeuristicReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_terms = [term.casefold() for term in query.split() if term.strip()]

        def _score(chunk: RetrievedChunk) -> float:
            lexical = sum(chunk.content.casefold().count(term) for term in query_terms)
            return chunk.score + (lexical * 0.05)

        ranked = sorted(chunks, key=_score, reverse=True)
        for index, chunk in enumerate(ranked, start=1):
            chunk.rank = index
        return ranked
