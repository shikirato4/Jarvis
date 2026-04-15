from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from .models import ResearchConflict, ResearchEvidence, ResearchFinding, ResearchFindingStatus, ResearchValidationResult


def validate_findings(task_id: str, findings: list[ResearchFinding], evidence: list[ResearchEvidence]) -> tuple[list[ResearchValidationResult], list[ResearchConflict]]:
    evidence_map = {item.evidence_id: item for item in evidence}
    topic_map: dict[str, list[ResearchFinding]] = defaultdict(list)
    validations: list[ResearchValidationResult] = []
    conflicts: list[ResearchConflict] = []

    for finding in findings:
        topic_map[finding.topic.casefold()].append(finding)
        supporting = [evidence_map[item] for item in finding.evidence_ids if item in evidence_map]
        sources = {item.source.source_id for item in supporting}
        valid = len(sources) >= 2 or len(supporting) >= 2
        confidence = min(0.35 + (0.2 * len(sources)) + (0.15 * len(supporting)), 0.95)
        validations.append(
            ResearchValidationResult(
                finding_id=finding.finding_id,
                valid=valid,
                confidence=round(confidence, 3),
                cross_checked_sources=len(sources),
                supporting_evidence_ids=[item.evidence_id for item in supporting],
                notes=["single_source"] if len(sources) <= 1 else [],
            )
        )

    for same_topic in topic_map.values():
        for index, left in enumerate(same_topic):
            left_norm = left.claim.casefold()
            for right in same_topic[index + 1 :]:
                right_norm = right.claim.casefold()
                if left_norm == right_norm:
                    continue
                if left_norm in right_norm or right_norm in left_norm:
                    continue
                left_tokens = set(left_norm.split())
                right_tokens = set(right_norm.split())
                overlap = left_tokens.intersection(right_tokens)
                if left_tokens and right_tokens and len(overlap) <= 2:
                    conflict_id = str(uuid4())
                    conflicts.append(
                        ResearchConflict(
                            conflict_id=conflict_id,
                            task_id=task_id,
                            topic=left.topic,
                            claim_a=left.claim,
                            claim_b=right.claim,
                            evidence_ids=list(dict.fromkeys([*left.evidence_ids, *right.evidence_ids])),
                            severity="medium",
                            resolution="requires manual review",
                        )
                    )
                    left.contradiction_ids.append(conflict_id)
                    right.contradiction_ids.append(conflict_id)
                    left.status = ResearchFindingStatus.CONTRADICTED
                    right.status = ResearchFindingStatus.CONTRADICTED

    return validations, conflicts
