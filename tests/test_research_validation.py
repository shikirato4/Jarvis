from jarvis.research_runtime.models import ResearchEvidence, ResearchFinding, ResearchSource, ResearchSourceKind
from jarvis.research_runtime.validation import validate_findings


def test_research_validation_detects_conflicts_and_single_source_weakness() -> None:
    source_a = ResearchSource(source_id="s1", kind=ResearchSourceKind.SIMULATED, provider_name="sim", display_name="A")
    source_b = ResearchSource(source_id="s2", kind=ResearchSourceKind.SIMULATED, provider_name="sim", display_name="B")
    evidence = [
        ResearchEvidence(evidence_id="e1", task_id="t1", source=source_a, content="Jarvis is modular.", excerpt="Jarvis is modular.", citation="A"),
        ResearchEvidence(evidence_id="e2", task_id="t1", source=source_b, content="Jarvis is monolithic.", excerpt="Jarvis is monolithic.", citation="B"),
    ]
    findings = [
        ResearchFinding(finding_id="f1", task_id="t1", topic="architecture", claim="Jarvis is modular", summary="Jarvis is modular", evidence_ids=["e1"], citations=["A"]),
        ResearchFinding(finding_id="f2", task_id="t1", topic="architecture", claim="Jarvis is monolithic", summary="Jarvis is monolithic", evidence_ids=["e2"], citations=["B"]),
    ]
    validations, conflicts = validate_findings("t1", findings, evidence)
    assert len(validations) == 2
    assert conflicts
    assert findings[0].contradiction_ids
