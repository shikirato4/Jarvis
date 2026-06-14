from __future__ import annotations

from jarvis.code_agent_runtime.patches.models import PatchReview, ProposedPatch


def review_patch(patch: ProposedPatch) -> PatchReview:
    warnings = list(patch.warnings)
    if patch.risk_level >= 2 and not patch.requires_confirmation:
        warnings.append("sensitive patch should require confirmation")
    if len(patch.unified_diff) > 20_000:
        warnings.append("diff is large and should be reviewed manually")
    return PatchReview(patch_id=patch.id, status="needs_review" if warnings else "ok", warnings=warnings, summary=patch.summary)
