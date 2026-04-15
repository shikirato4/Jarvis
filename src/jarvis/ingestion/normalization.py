from __future__ import annotations

import hashlib
from pathlib import Path

from jarvis.memory_semantic.documents import DocumentProvenance


class NormalizedDocument:
    def __init__(self, *, title: str, content: str, provenance: DocumentProvenance) -> None:
        self.title = title
        self.content = content
        self.provenance = provenance


def normalize_text_content(content: str) -> str:
    lines = [line.rstrip() for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        blank = not line.strip()
        if blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = blank
    return "\n".join(compacted).strip()


def build_provenance(*, source_path: Path | None, content: str, author: str | None = None, section: str | None = None) -> DocumentProvenance:
    return DocumentProvenance(
        source_path=str(source_path) if source_path else None,
        checksum_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        author=author,
        section=section,
    )
