from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from jarvis.config import Settings
from jarvis.ingestion.loaders import DocumentLoader
from jarvis.ingestion.normalization import build_provenance, normalize_text_content
from jarvis.memory_semantic.documents import SourceType
from jarvis.research_runtime.models import ResearchTaskStatus
from jarvis.writing_runtime.models import WritingTaskStatus

from .models import DiscoveredSourceItem, IndexDocumentType, IndexSource, IndexSourceKind, IndexedDocument
from .safeguards import ensure_safe_scan, is_allowed_extension, is_sensitive_path


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class IndexingIngestionService:
    def __init__(self, settings: Settings, research_repository, writing_repository) -> None:
        self._settings = settings
        self._research_repository = research_repository
        self._writing_repository = writing_repository

    def discover(self, source: IndexSource) -> list[DiscoveredSourceItem]:
        if source.source_kind in {
            IndexSourceKind.WORKSPACE_FILES,
            IndexSourceKind.USER_DOCUMENTS,
            IndexSourceKind.CODE_PROJECT,
            IndexSourceKind.UNITY_PROJECT,
        }:
            return self._discover_files(source)
        if source.source_kind == IndexSourceKind.RESEARCH_RESULTS:
            return self._discover_research(source)
        if source.source_kind == IndexSourceKind.WRITING_ARTIFACTS:
            return self._discover_writing(source)
        return []

    def load_document(self, item: DiscoveredSourceItem, source: IndexSource, *, snapshot_id: str, existing_document_id: str | None = None) -> IndexedDocument:
        content = normalize_text_content(item.content)
        fingerprint = _hash_text(f"{item.canonical_uri}:{item.content_hash}:{content}")
        return IndexedDocument(
            document_id=existing_document_id or item.content_hash[:36],
            source_id=source.source_id,
            snapshot_id=snapshot_id,
            canonical_uri=item.canonical_uri,
            path=item.path,
            title=item.title,
            document_type=item.document_type,
            mime_type=str(item.metadata.get("mime_type")) if item.metadata.get("mime_type") else None,
            language=str(item.metadata.get("language") or "unknown"),
            fingerprint=fingerprint,
            content_hash=item.content_hash,
            source_version=item.modified_at.isoformat() if item.modified_at else None,
            size_bytes=item.size_bytes,
            char_count=len(content),
            token_estimate=max(len(content.split()), 1) if content else 0,
            content=content,
            metadata=item.metadata.copy(),
            provenance=item.provenance.copy(),
            is_sensitive=bool(item.metadata.get("is_sensitive", False)),
        )

    def _discover_files(self, source: IndexSource) -> list[DiscoveredSourceItem]:
        root = ensure_safe_scan(source, self._settings)
        assert root is not None
        loader = DocumentLoader((root,))
        items: list[DiscoveredSourceItem] = []
        excluded_dirnames = {name.casefold() for name in self._settings.indexing_excluded_dirnames}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(root).parts
            if any(self._is_excluded_dir_part(part, excluded_dirnames) for part in relative_parts[:-1]):
                continue
            if is_sensitive_path(path, self._settings, source):
                continue
            if not is_allowed_extension(path, self._settings, source):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > (source.max_file_size_bytes or self._settings.indexing_max_file_size_bytes):
                continue
            document_type = self._infer_document_type(path, source)
            content = self._load_file_content(path, loader, document_type)
            if not content.strip():
                continue
            items.append(
                DiscoveredSourceItem(
                    source_id=source.source_id,
                    source_kind=source.source_kind,
                    canonical_uri=path.resolve().as_uri(),
                    path=str(path.resolve()),
                    title=path.stem,
                    document_type=document_type,
                    content=content,
                    content_hash=_hash_text(content),
                    metadata={
                        "root_path": str(root),
                        "extension": path.suffix.lower(),
                        "mime_type": "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain",
                        "project_hint": source.source_kind.value,
                    },
                    provenance=build_provenance(source_path=path.resolve(), content=content).model_dump(mode="json"),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return items

    @staticmethod
    def _is_excluded_dir_part(part: str, excluded_dirnames: set[str]) -> bool:
        lowered = part.casefold()
        return lowered in excluded_dirnames or lowered.startswith(".venv")

    def _discover_research(self, source: IndexSource) -> list[DiscoveredSourceItem]:
        items: list[DiscoveredSourceItem] = []
        for task in self._research_repository.list_tasks(limit=100):
            if task.status not in {ResearchTaskStatus.COMPLETED, ResearchTaskStatus.DELEGATED, ResearchTaskStatus.RUNNING} and task.report is None:
                continue
            if task.report is None:
                continue
            report = task.report
            content = "\n\n".join(
                [
                    report.title,
                    report.short_summary,
                    report.detailed_summary,
                    report.technical_analysis,
                    "\n".join(report.key_points),
                ]
            ).strip()
            items.append(
                DiscoveredSourceItem(
                    source_id=source.source_id,
                    source_kind=source.source_kind,
                    canonical_uri=f"research://{task.task_id}",
                    title=report.title,
                    document_type=IndexDocumentType.RESEARCH_REPORT,
                    content=content,
                    content_hash=_hash_text(content),
                    metadata={"task_id": task.task_id, "report_id": report.report_id, "kind": "research_report"},
                    provenance={"task_id": task.task_id, "report_id": report.report_id},
                    modified_at=task.updated_at,
                )
            )
        return items

    def _discover_writing(self, source: IndexSource) -> list[DiscoveredSourceItem]:
        items: list[DiscoveredSourceItem] = []
        for task in self._writing_repository.list_tasks(limit=100):
            if task.status not in {WritingTaskStatus.COMPLETED, WritingTaskStatus.DELEGATED, WritingTaskStatus.RUNNING} and not task.generated_blocks:
                continue
            content = "\n\n".join(
                filter(
                    None,
                    [
                        task.goal,
                        task.context.combined_context,
                        "\n".join(block.text for block in task.generated_blocks),
                    ],
                )
            ).strip()
            if not content:
                continue
            items.append(
                DiscoveredSourceItem(
                    source_id=source.source_id,
                    source_kind=source.source_kind,
                    canonical_uri=f"writing://{task.task_id}",
                    title=task.document_title or task.goal[:120],
                    document_type=IndexDocumentType.WRITING_CONTEXT,
                    content=content,
                    content_hash=_hash_text(content),
                    metadata={"task_id": task.task_id, "application_name": task.application_name, "kind": "writing_context"},
                    provenance={"task_id": task.task_id, "document_title": task.document_title},
                    modified_at=task.updated_at,
                )
            )
        return items

    @staticmethod
    def _infer_document_type(path: Path, source: IndexSource) -> IndexDocumentType:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return IndexDocumentType.MARKDOWN
        if suffix == ".json":
            return IndexDocumentType.JSON
        if suffix == ".pdf":
            return IndexDocumentType.PDF
        if suffix in {".cs", ".py", ".js", ".ts", ".tsx", ".jsx", ".cpp", ".h", ".hpp"}:
            return IndexDocumentType.CODE
        if source.source_kind == IndexSourceKind.UNITY_PROJECT:
            return IndexDocumentType.UNITY_ASSET
        return IndexDocumentType.TEXT

    def _load_file_content(self, path: Path, loader: DocumentLoader, document_type: IndexDocumentType) -> str:
        if document_type == IndexDocumentType.JSON:
            return loader.load(str(path), source_type=SourceType.JSON).content
        if document_type == IndexDocumentType.MARKDOWN:
            return loader.load(str(path), source_type=SourceType.MARKDOWN).content
        if document_type == IndexDocumentType.PDF:
            try:
                import pypdf  # type: ignore

                reader = pypdf.PdfReader(str(path))
                return normalize_text_content("\n\n".join(page.extract_text() or "" for page in reader.pages))
            except Exception:
                return ""
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return normalize_text_content(json.dumps(payload, indent=2, ensure_ascii=False))
        return normalize_text_content(path.read_text(encoding="utf-8", errors="ignore"))
