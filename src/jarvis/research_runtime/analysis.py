from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from .base import ResearchModelAdapter
from .models import ResearchEvidence, ResearchFinding, ResearchFindingStatus, ResearchSupportLevel, ResearchTask


class ResearchAnalyzer:
    def __init__(self, models: ResearchModelAdapter) -> None:
        self._models = models

    def expand_queries(self, task: ResearchTask, *, correlation_id: str) -> tuple[list[str], list[str]]:
        prompt = (
            "Expand this research query into alternative queries and hypotheses.\n"
            f"Query: {task.query}\n"
            'Return JSON with keys "queries" and "hypotheses".'
        )
        response = self._models.infer_json(
            task_type="analysis",
            logical_model="planner",
            prompt=prompt,
            correlation_id=correlation_id,
            metadata={"component": "research_runtime", "stage": "query_expansion"},
        )
        if response:
            queries = [str(item) for item in response.get("queries", []) if str(item).strip()]
            hypotheses = [str(item) for item in response.get("hypotheses", []) if str(item).strip()]
            if queries or hypotheses:
                return queries[:6], hypotheses[:6]
        base_terms = [task.query]
        if " " in task.query:
            base_terms.append(task.query.replace("?", ""))
        return list(dict.fromkeys(base_terms))[:4], [f"Hypothesis: {task.query} has multiple corroborating sources."]

    def analyze(self, task: ResearchTask, evidence: list[ResearchEvidence], *, correlation_id: str) -> list[ResearchFinding]:
        if not evidence:
            return []
        prompt = (
            "Analyze the evidence and extract findings.\n"
            f"Query: {task.query}\n"
            "Evidence:\n"
            + "\n".join(f"- [{item.source.display_name}] {item.excerpt}" for item in evidence[:12])
            + '\nReturn JSON with key "findings" as a list of objects with topic, claim, summary, evidence_indexes.'
        )
        response = self._models.infer_json(
            task_type="analysis",
            logical_model="general_assistant",
            prompt=prompt,
            correlation_id=correlation_id,
            metadata={"component": "research_runtime", "stage": "analysis"},
        )
        findings: list[ResearchFinding] = []
        if response and isinstance(response.get("findings"), list):
            for item in response["findings"]:
                if not isinstance(item, dict):
                    continue
                indexes = item.get("evidence_indexes", [])
                selected = [evidence[idx] for idx in indexes if isinstance(idx, int) and 0 <= idx < len(evidence)]
                if not selected:
                    continue
                findings.append(
                    ResearchFinding(
                        finding_id=str(uuid4()),
                        task_id=task.task_id,
                        topic=str(item.get("topic") or "general"),
                        claim=str(item.get("claim") or selected[0].excerpt),
                        summary=str(item.get("summary") or selected[0].excerpt),
                        evidence_ids=[entry.evidence_id for entry in selected],
                        citations=[entry.citation for entry in selected],
                        status=ResearchFindingStatus.ANALYZED,
                        support_level=ResearchSupportLevel.MEDIUM,
                    )
                )
        if findings:
            return findings[: task.budget.max_findings]
        return self._fallback_findings(task, evidence)

    def _fallback_findings(self, task: ResearchTask, evidence: list[ResearchEvidence]) -> list[ResearchFinding]:
        grouped: dict[str, list[ResearchEvidence]] = defaultdict(list)
        for item in evidence:
            topic = str(item.metadata.get("topic") or item.source.kind.value)
            grouped[topic].append(item)
        findings: list[ResearchFinding] = []
        for topic, entries in grouped.items():
            excerpts = [entry.excerpt for entry in entries[:2]]
            claim = " ".join(excerpts)
            findings.append(
                ResearchFinding(
                    finding_id=str(uuid4()),
                    task_id=task.task_id,
                    topic=topic,
                    claim=claim,
                    summary=claim[:280],
                    evidence_ids=[entry.evidence_id for entry in entries[:3]],
                    citations=[entry.citation for entry in entries[:3]],
                    status=ResearchFindingStatus.ANALYZED,
                    support_level=ResearchSupportLevel.MEDIUM if len(entries) > 1 else ResearchSupportLevel.LOW,
                )
            )
        return findings[: task.budget.max_findings]
