from __future__ import annotations

from uuid import uuid4

from jarvis.core.safety import ensure_within_roots
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.vision_runtime.base import OCRRequest

from .models import ResearchEvidence, ResearchRunRequest, ResearchSource, ResearchSourceKind, ResearchTask


class ResearchRetriever:
    def __init__(self, settings, semantic_memory, memory_service, vision_runtime, system_runtime, logger=None) -> None:
        self._settings = settings
        self._semantic_memory = semantic_memory
        self._memory_service = memory_service
        self._vision_runtime = vision_runtime
        self._system_runtime = system_runtime
        self._logger = logger

    def retrieve(self, task: ResearchTask, request: ResearchRunRequest, *, correlation_id: str) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        if "semantic_memory" in task.source_scope:
            found_sources, found_evidence = self._from_semantic_memory(task, correlation_id=correlation_id)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        if "workspace" in task.source_scope:
            found_sources, found_evidence = self._from_workspace(task)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        if "simulated" in task.source_scope and request.simulated_sources:
            found_sources, found_evidence = self._from_simulated(task, request)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        if task.paths:
            found_sources, found_evidence = self._from_files(task)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        if task.image_paths:
            found_sources, found_evidence = self._from_images(task, correlation_id=correlation_id)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        if not evidence:
            found_sources, found_evidence = self._from_memory(task)
            sources.extend(found_sources)
            evidence.extend(found_evidence)
        limited_sources = sources[: task.budget.max_sources]
        source_ids = {item.source_id for item in limited_sources}
        limited_evidence = [item for item in evidence if item.source.source_id in source_ids][: task.budget.max_sources * 4]
        return limited_sources, limited_evidence

    def _from_semantic_memory(self, task: ResearchTask, *, correlation_id: str) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        try:
            context = self._semantic_memory.retrieve_context(
                SemanticSearchQuery(
                    query=task.query,
                    collection_name=task.collection_name,
                    top_k=min(task.budget.max_sources * 2, 10),
                    correlation_id=correlation_id,
                )
            )
        except Exception:
            return [], []
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        for chunk in context.chunks:
            source = ResearchSource(
                source_id=f"semantic-{chunk.chunk_id}",
                kind=ResearchSourceKind.SEMANTIC_MEMORY,
                provider_name="semantic_memory",
                display_name=chunk.document_title or chunk.collection_name,
                location=chunk.source_path,
                trust_level="high",
                freshness="persisted",
                metadata={"collection_name": chunk.collection_name, "rank": chunk.rank, "score": chunk.score},
                capabilities=("retrieve", "cite"),
            )
            sources.append(source)
            evidence.append(
                ResearchEvidence(
                    evidence_id=str(uuid4()),
                    task_id=task.task_id,
                    source=source,
                    content=chunk.content,
                    excerpt=chunk.content[:280],
                    citation=chunk.source_path or f"{chunk.collection_name}:{chunk.chunk_id}",
                    metadata={"topic": chunk.collection_name, "semantic_score": chunk.score},
                    score=chunk.score,
                )
            )
        return sources, evidence

    def _from_workspace(self, task: ResearchTask) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        allowed_roots = self._settings.resolved_research_roots
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        query_tokens = {token.casefold() for token in task.query.split() if len(token) > 2}
        for root in allowed_roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {item.lower() for item in self._settings.research_default_extensions}:
                    continue
                try:
                    if path.stat().st_size > self._settings.research_max_file_size_bytes:
                        continue
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                matches = [line for line in lines if any(token in line.casefold() for token in query_tokens)]
                if not matches:
                    continue
                source = ResearchSource(
                    source_id=f"workspace-{path.as_posix()}",
                    kind=ResearchSourceKind.WORKSPACE,
                    provider_name="workspace_search",
                    display_name=path.name,
                    location=str(path),
                    trust_level="medium",
                    freshness="local",
                    metadata={"path": str(path)},
                    capabilities=("search", "cite"),
                )
                sources.append(source)
                for match in matches[:2]:
                    evidence.append(
                        ResearchEvidence(
                            evidence_id=str(uuid4()),
                            task_id=task.task_id,
                            source=source,
                            content=text[:2_000],
                            excerpt=match[:280],
                            citation=str(path),
                            metadata={"topic": path.suffix.lower()},
                            score=0.55,
                        )
                    )
        return sources, evidence

    def _from_simulated(self, task: ResearchTask, request: ResearchRunRequest) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        for index, item in enumerate(request.simulated_sources, start=1):
            source = ResearchSource(
                source_id=f"simulated-{index}",
                kind=item.kind,
                provider_name="simulated_source",
                display_name=item.title,
                location=item.location,
                trust_level="medium",
                freshness="test",
                metadata=item.metadata,
                capabilities=("search", "cite"),
            )
            sources.append(source)
            evidence.append(
                ResearchEvidence(
                    evidence_id=str(uuid4()),
                    task_id=task.task_id,
                    source=source,
                    content=item.content,
                    excerpt=item.content[:280],
                    citation=item.location or item.title,
                    metadata={"topic": "simulated"},
                    score=0.65,
                )
            )
        return sources, evidence

    def _from_files(self, task: ResearchTask) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        for raw_path in task.paths:
            try:
                path = ensure_within_roots(raw_path, self._settings.resolved_research_roots, "research file input")
            except Exception:
                try:
                    resolved = self._system_runtime.resolve({"query": raw_path, "metadata": {"component": "research_runtime"}})
                    if resolved.resolved_target.path is None:
                        continue
                    path = ensure_within_roots(resolved.resolved_target.path, self._settings.resolved_research_roots, "research file input")
                except Exception:
                    continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            kind = ResearchSourceKind.PDF if path.suffix.lower() == ".pdf" else ResearchSourceKind.FILE
            source = ResearchSource(
                source_id=f"file-{path.as_posix()}",
                kind=kind,
                provider_name="local_file",
                display_name=path.name,
                location=str(path),
                trust_level="high",
                freshness="local",
                metadata={"path": str(path)},
                capabilities=("read", "cite"),
            )
            sources.append(source)
            evidence.append(
                ResearchEvidence(
                    evidence_id=str(uuid4()),
                    task_id=task.task_id,
                    source=source,
                    content=text,
                    excerpt=text[:280],
                    citation=str(path),
                    metadata={"topic": path.suffix.lower()},
                    score=0.7,
                )
            )
        return sources, evidence

    def _from_images(self, task: ResearchTask, *, correlation_id: str) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        for raw_path in task.image_paths:
            try:
                path = ensure_within_roots(raw_path, self._settings.resolved_research_roots, "research image input")
            except Exception:
                continue
            try:
                receipt = self._vision_runtime.extract_text(OCRRequest(image_path=str(path), correlation_id=correlation_id))
            except Exception:
                continue
            text = receipt.ocr_result.text if receipt.ocr_result else ""
            if not text.strip():
                continue
            source = ResearchSource(
                source_id=f"image-{path.as_posix()}",
                kind=ResearchSourceKind.IMAGE,
                provider_name="vision_runtime",
                display_name=path.name,
                location=str(path),
                trust_level="medium",
                freshness="local",
                metadata={"path": str(path), "operation": receipt.operation_name},
                capabilities=("ocr", "cite"),
            )
            sources.append(source)
            evidence.append(
                ResearchEvidence(
                    evidence_id=str(uuid4()),
                    task_id=task.task_id,
                    source=source,
                    content=text,
                    excerpt=text[:280],
                    citation=str(path),
                    metadata={"topic": "ocr"},
                    score=0.6,
                )
            )
        return sources, evidence

    def _from_memory(self, task: ResearchTask) -> tuple[list[ResearchSource], list[ResearchEvidence]]:
        matches = self._memory_service.search_memories(task.query, limit=5)
        sources: list[ResearchSource] = []
        evidence: list[ResearchEvidence] = []
        for item in matches:
            source = ResearchSource(
                source_id=f"memory-{item.id}",
                kind=ResearchSourceKind.MEMORY,
                provider_name="memory_service",
                display_name=f"memory:{item.kind}",
                location=None,
                trust_level="medium",
                freshness="historic",
                metadata={"kind": item.kind, "created_at": item.created_at.isoformat()},
                capabilities=("search",),
            )
            sources.append(source)
            evidence.append(
                ResearchEvidence(
                    evidence_id=str(uuid4()),
                    task_id=task.task_id,
                    source=source,
                    content=item.content,
                    excerpt=item.content[:280],
                    citation=f"memory:{item.id}",
                    metadata={"topic": item.kind},
                    score=0.45,
                )
            )
        return sources, evidence
