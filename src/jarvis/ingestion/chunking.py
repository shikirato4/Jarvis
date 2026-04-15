from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TextChunk:
    index: int
    content: str
    char_count: int
    token_estimate: int


class TextChunker:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []
        chunks: list[TextChunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            segment = text[start:end].strip()
            if segment:
                chunks.append(
                    TextChunk(
                        index=index,
                        content=segment,
                        char_count=len(segment),
                        token_estimate=max(len(segment.split()), 1),
                    )
                )
                index += 1
            if end >= len(text):
                break
            start = max(end - self._chunk_overlap, start + 1)
        return chunks
