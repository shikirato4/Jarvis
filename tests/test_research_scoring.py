from jarvis.research_runtime.models import (
    ResearchFinding,
    ResearchFindingStatus,
    ResearchSupportLevel,
    ResearchValidationResult,
)
from jarvis.research_runtime.scoring import confidence_level, score_finding


def test_research_scoring_weights_evidence_and_consistency() -> None:
    finding = ResearchFinding(
        finding_id="f1",
        task_id="t1",
        topic="runtime",
        claim="The runtime is modular.",
        summary="The runtime is modular and service-based.",
        evidence_ids=["e1", "e2", "e3"],
        citations=["a", "b"],
        status=ResearchFindingStatus.VERIFIED,
        support_level=ResearchSupportLevel.HIGH,
    )
    validation = ResearchValidationResult(finding_id="f1", valid=True, confidence=0.9, cross_checked_sources=2)
    score = score_finding(finding, validation)
    assert score.overall >= 0.8
    assert confidence_level(score.overall) == "high"
