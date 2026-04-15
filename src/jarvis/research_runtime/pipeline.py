from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from jarvis.memory_semantic.documents import DocumentIngestionRequest

from .analysis import ResearchAnalyzer
from .models import ResearchFindingStatus, ResearchRunRequest, ResearchStep, ResearchStepKind, ResearchStepStatus, ResearchTask, ResearchTaskStatus
from .retrieval import ResearchRetriever
from .safeguards import budget_exceeded
from .scoring import score_finding
from .synthesis import ResearchSynthesizer
from .validation import validate_findings


class ResearchPipeline:
    def __init__(self, *, retriever: ResearchRetriever, analyzer: ResearchAnalyzer, synthesizer: ResearchSynthesizer, semantic_memory, logger=None) -> None:
        self._retriever = retriever
        self._analyzer = analyzer
        self._synthesizer = synthesizer
        self._semantic_memory = semantic_memory
        self._logger = logger

    def run(self, task: ResearchTask, request: ResearchRunRequest, *, correlation_id: str, operation_handle=None) -> ResearchTask:
        started_at = datetime.now(timezone.utc)
        model_calls = 0
        task.status = ResearchTaskStatus.RUNNING
        task.updated_at = datetime.now(timezone.utc)
        self._check_cancelled(operation_handle, task)

        self._record_step(task, ResearchStepKind.QUERY_PARSING, "Parse query", {"query": task.query}, {"normalized_query": task.query.strip()})
        self._heartbeat(operation_handle, "query parsing completed")

        self._check_cancelled(operation_handle, task)
        queries, hypotheses = self._analyzer.expand_queries(task, correlation_id=correlation_id)
        model_calls += 1
        task.expanded_queries = list(dict.fromkeys([task.query, *queries]))[:6]
        task.hypotheses = hypotheses
        self._record_step(task, ResearchStepKind.QUERY_EXPANSION, "Expand query", {"query": task.query}, {"expanded_queries": task.expanded_queries, "hypotheses": hypotheses})
        self._heartbeat(operation_handle, "query expansion completed")

        self._check_cancelled(operation_handle, task)
        sources, evidence = self._retriever.retrieve(task, request, correlation_id=correlation_id)
        task.sources = sources
        task.evidence = evidence
        self._record_step(task, ResearchStepKind.RETRIEVAL, "Retrieve evidence", {"query": task.query}, {"source_count": len(sources), "evidence_count": len(evidence)})
        self._heartbeat(operation_handle, "retrieval completed")

        self._check_cancelled(operation_handle, task)
        findings = self._analyzer.analyze(task, evidence, correlation_id=correlation_id)
        model_calls += 1
        task.findings = findings
        self._record_step(task, ResearchStepKind.ANALYSIS, "Analyze evidence", {"evidence_count": len(evidence)}, {"finding_count": len(findings)})
        self._heartbeat(operation_handle, "analysis completed")

        self._check_cancelled(operation_handle, task)
        validations, conflicts = validate_findings(task.task_id, findings, evidence)
        task.validations = validations
        task.conflicts = conflicts
        for finding in task.findings:
            validation = next((item for item in validations if item.finding_id == finding.finding_id), None)
            score = score_finding(finding, validation)
            finding.confidence = score.overall
            if validation and validation.valid:
                finding.status = ResearchFindingStatus.VERIFIED if not finding.contradiction_ids else ResearchFindingStatus.CONTRADICTED
            elif finding.status != ResearchFindingStatus.CONTRADICTED:
                finding.status = ResearchFindingStatus.WEAKLY_SUPPORTED
            finding.support_level = "high" if score.overall >= 0.8 else "medium" if score.overall >= 0.55 else "low"
            finding.metadata["score"] = score.model_dump(mode="json")
        self._record_step(task, ResearchStepKind.CROSS_VALIDATION, "Cross validate findings", {"finding_count": len(findings)}, {"validation_count": len(validations)})
        self._record_step(task, ResearchStepKind.CONFLICT_DETECTION, "Detect conflicts", {"finding_count": len(findings)}, {"conflict_count": len(conflicts)})
        self._heartbeat(operation_handle, "validation completed")

        self._check_cancelled(operation_handle, task)
        confidence = round(sum(item.confidence for item in task.findings) / len(task.findings), 3) if task.findings else 0.0
        task.report = self._synthesizer.synthesize(task, task.findings, task.conflicts, confidence=confidence, correlation_id=correlation_id)
        self._record_step(task, ResearchStepKind.SYNTHESIS, "Synthesize report", {"finding_count": len(task.findings)}, {"confidence": confidence})
        self._record_step(task, ResearchStepKind.REPORT_GENERATION, "Generate report", {"task_id": task.task_id}, {"report_id": task.report.report_id})

        if request.persist_results and task.report is not None:
            self._persist_report(task)
        self._record_step(task, ResearchStepKind.PERSIST_RESULTS, "Persist results", {"task_id": task.task_id}, {"persisted": request.persist_results})
        self._heartbeat(operation_handle, "persistence completed")

        violation = budget_exceeded(task, model_calls=model_calls, started_at=started_at)
        if violation:
            task.status = ResearchTaskStatus.FAILED
            task.last_error = violation
        else:
            task.status = ResearchTaskStatus.COMPLETED
        task.updated_at = datetime.now(timezone.utc)
        return task

    @staticmethod
    def _heartbeat(operation_handle, progress_message: str) -> None:
        if operation_handle is not None:
            operation_handle.heartbeat(progress_message=progress_message)

    @staticmethod
    def _check_cancelled(operation_handle, task: ResearchTask) -> None:
        if operation_handle is not None:
            operation_handle.raise_if_cancelled(component="research_runtime")
        if task.status == ResearchTaskStatus.CANCELLED:
            raise RuntimeError("research task cancelled")

    def _persist_report(self, task: ResearchTask) -> None:
        if task.report is None:
            return
        settings = getattr(self._semantic_memory, "_settings", None)
        if settings is not None and (not getattr(settings, "ollama_enabled", True) or not getattr(settings, "embeddings_enabled", True)):
            return
        try:
            self._semantic_memory.ingest_document(
                DocumentIngestionRequest(
                    collection_name=task.collection_name or "deep_research",
                    source_type="research_note",
                    content=task.report.detailed_summary,
                    title=task.report.title,
                    metadata={"task_id": task.task_id, "report_id": task.report.report_id, "kind": "deep_research_report"},
                    persist_memory=True,
                )
            )
        except Exception:
            return

    @staticmethod
    def _record_step(task: ResearchTask, kind: ResearchStepKind, title: str, input_payload: dict, output_payload: dict) -> None:
        now = datetime.now(timezone.utc)
        task.steps.append(
            ResearchStep(
                step_id=f"{kind.value}-{uuid4().hex[:8]}",
                task_id=task.task_id,
                kind=kind,
                title=title,
                description=title,
                status=ResearchStepStatus.COMPLETED,
                input_payload=input_payload,
                output_payload=output_payload,
                started_at=now,
                finished_at=now,
            )
        )
