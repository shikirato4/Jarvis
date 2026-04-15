from __future__ import annotations

from datetime import datetime, timezone

from .models import ResearchBudget, ResearchTask


def should_require_approval(task: ResearchTask) -> bool:
    return task.autonomy_enabled and (
        task.budget.max_duration_seconds >= 180
        or task.budget.max_sources >= 8
        or len(task.paths) + len(task.image_paths) >= 5
    )


def budget_exceeded(task: ResearchTask, *, model_calls: int, started_at: datetime) -> str | None:
    if len(task.steps) > task.budget.max_steps:
        return "max_steps_exceeded"
    if model_calls > task.budget.max_model_calls:
        return "max_model_calls_exceeded"
    if (datetime.now(timezone.utc) - started_at).total_seconds() > task.budget.max_duration_seconds:
        return "max_duration_exceeded"
    if len(task.sources) > task.budget.max_sources:
        return "max_sources_exceeded"
    if len(task.findings) > task.budget.max_findings:
        return "max_findings_exceeded"
    return None


def merge_budget(base: ResearchBudget, override: ResearchBudget | None) -> ResearchBudget:
    if override is None:
        return base
    return override
