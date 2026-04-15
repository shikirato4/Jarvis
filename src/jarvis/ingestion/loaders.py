from __future__ import annotations

import json
from pathlib import Path

from jarvis.core.errors import ConfigurationError
from jarvis.core.safety import ensure_within_roots
from jarvis.memory_semantic.documents import SourceType

from .normalization import NormalizedDocument, build_provenance, normalize_text_content


class DocumentLoader:
    def __init__(self, allowed_roots: tuple[Path, ...]) -> None:
        self._allowed_roots = allowed_roots

    def load(self, path: str, *, source_type: SourceType, title: str | None = None) -> NormalizedDocument:
        resolved = ensure_within_roots(path, self._allowed_roots, "semantic ingestion")
        suffix = resolved.suffix.lower()
        if source_type == SourceType.JSON or suffix == ".json":
            content = self._load_json(resolved)
        elif source_type in {SourceType.MARKDOWN, SourceType.TEXT, SourceType.NOTE, SourceType.RESEARCH_NOTE, SourceType.DRAFT, SourceType.BOOK} or suffix in {
            ".md",
            ".txt",
            ".py",
            ".toml",
            ".yaml",
            ".yml",
            ".ini",
        }:
            content = resolved.read_text(encoding="utf-8")
        else:
            raise ConfigurationError(f"source type '{source_type.value}' is not supported yet")
        normalized = normalize_text_content(content)
        return NormalizedDocument(
            title=title or resolved.stem,
            content=normalized,
            provenance=build_provenance(source_path=resolved, content=normalized),
        )

    @staticmethod
    def _load_json(path: Path) -> str:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return normalize_text_content(json.dumps(payload, indent=2, ensure_ascii=False))
