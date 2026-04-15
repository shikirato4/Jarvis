from __future__ import annotations

from .base import ResearchModelAdapter
from .models import ResearchConflict, ResearchFinding, ResearchReport, ResearchTask
from .scoring import confidence_level


class ResearchSynthesizer:
    def __init__(self, models: ResearchModelAdapter) -> None:
        self._models = models

    def synthesize(
        self,
        task: ResearchTask,
        findings: list[ResearchFinding],
        conflicts: list[ResearchConflict],
        *,
        confidence: float,
        correlation_id: str,
    ) -> ResearchReport:
        short_summary = self._summarize_short(task, findings)
        detailed_summary = self._summarize_detailed(task, findings, conflicts)
        technical_analysis = self._technical_analysis(task, findings, conflicts, confidence)
        key_points = [item.summary for item in findings[:5]]
        structured = {
            "query": task.query,
            "expanded_queries": task.expanded_queries,
            "hypotheses": task.hypotheses,
            "findings": [item.model_dump(mode="json") for item in findings],
            "conflicts": [item.model_dump(mode="json") for item in conflicts],
            "confidence": confidence,
            "sources": [item.model_dump(mode="json") for item in task.sources],
        }
        return ResearchReport(
            report_id=f"report-{task.task_id}",
            task_id=task.task_id,
            title=f"Deep research report: {task.query}",
            short_summary=short_summary,
            detailed_summary=detailed_summary,
            technical_analysis=technical_analysis,
            key_points=key_points,
            hypotheses=task.hypotheses,
            findings=findings,
            conflicts=conflicts,
            confidence=confidence,
            confidence_level=confidence_level(confidence),
            structured_report=structured,
            metadata={"requested_outputs": list(task.requested_outputs), "model_correlation_id": correlation_id},
        )

    @staticmethod
    def _summarize_short(task: ResearchTask, findings: list[ResearchFinding]) -> str:
        if not findings:
            return f"No robust findings were produced for '{task.query}'."
        return f"{task.query}: {findings[0].summary}"

    @staticmethod
    def _summarize_detailed(task: ResearchTask, findings: list[ResearchFinding], conflicts: list[ResearchConflict]) -> str:
        lines = [f"Investigation topic: {task.query}", "", "Findings:"]
        if findings:
            lines.extend(f"- {item.summary}" for item in findings[:8])
        else:
            lines.append("- No findings available.")
        lines.extend(["", "Conflicts:"])
        if conflicts:
            lines.extend(f"- {item.claim_a} <> {item.claim_b}" for item in conflicts[:5])
        else:
            lines.append("- No direct contradictions detected.")
        return "\n".join(lines)

    @staticmethod
    def _technical_analysis(task: ResearchTask, findings: list[ResearchFinding], conflicts: list[ResearchConflict], confidence: float) -> str:
        lines = [
            f"Research task id: {task.task_id}",
            f"Query: {task.query}",
            f"Expanded queries: {', '.join(task.expanded_queries) if task.expanded_queries else 'none'}",
            f"Evidence count: {len(task.evidence)}",
            f"Finding count: {len(findings)}",
            f"Conflict count: {len(conflicts)}",
            f"Confidence: {confidence:.3f}",
        ]
        return "\n".join(lines)
