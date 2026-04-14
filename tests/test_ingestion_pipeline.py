from __future__ import annotations

from jarvis.ingestion.chunking import TextChunker
from jarvis.ingestion.normalization import normalize_text_content


def test_chunker_preserves_overlap_and_multiple_chunks() -> None:
    text = "A" * 70 + "B" * 70 + "C" * 70
    chunks = TextChunker(chunk_size=90, chunk_overlap=20).split(text)
    assert len(chunks) >= 3
    assert chunks[0].content[-20:] == chunks[1].content[:20]


def test_normalization_compacts_duplicate_blank_lines() -> None:
    content = "Title\r\n\r\n\r\nBody line 1\r\n\r\nBody line 2\r\n"
    normalized = normalize_text_content(content)
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Title")
