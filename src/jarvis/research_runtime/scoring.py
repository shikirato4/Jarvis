from __future__ import annotations

from .models import ResearchFinding, ResearchScore, ResearchValidationResult


def score_finding(finding: ResearchFinding, validation: ResearchValidationResult | None) -> ResearchScore:
    evidence_strength = min(len(finding.evidence_ids) / 3, 1.0)
    citation_coverage = 1.0 if finding.citations else 0.0
    consistency = validation.confidence if validation else 0.0
    relevance = min(max(len(finding.summary) / 180, 0.2), 1.0)
    freshness = 0.8 if finding.metadata.get("fresh") else 0.5
    overall = round((evidence_strength * 0.3) + (citation_coverage * 0.2) + (consistency * 0.3) + (relevance * 0.1) + (freshness * 0.1), 3)
    return ResearchScore(
        relevance=round(relevance, 3),
        evidence_strength=round(evidence_strength, 3),
        consistency=round(consistency, 3),
        citation_coverage=round(citation_coverage, 3),
        freshness=round(freshness, 3),
        overall=overall,
        rationale="weighted evidence, citations and cross-source consistency",
    )


def confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"
