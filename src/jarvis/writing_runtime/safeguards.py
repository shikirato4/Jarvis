from __future__ import annotations

from jarvis.core.errors import UIValidationError

from .models import WritingContext, WritingStyleProfile, WritingTask, WritingTextType

SENSITIVE_WINDOW_TOKENS = ("terminal", "powershell", "cmd", "registro", "credential", "password")
EDITABLE_WINDOW_TOKENS = ("word", "document", "editor", "docs", "notion", "writer", "browser")


def validate_target_window(context: WritingContext, *, expected: str | None = None) -> None:
    title = (context.window_title or "").casefold()
    expected_lower = (expected or "").casefold()
    if not title:
        raise UIValidationError("active window could not be identified")
    if any(token in title for token in SENSITIVE_WINDOW_TOKENS):
        raise UIValidationError("active window is sensitive and blocked", details={"window_title": context.window_title})
    if expected and expected_lower not in title:
        is_word_match = "word" in expected_lower and "word" in title and context.application_name == "word"
        if not is_word_match:
            raise UIValidationError("active window does not match target", details={"window_title": context.window_title, "expected": expected})
    if not any(token in title for token in EDITABLE_WINDOW_TOKENS):
        raise UIValidationError("active window is not recognized as editable", details={"window_title": context.window_title})


def ensure_context_sufficiency(context: WritingContext) -> None:
    if len((context.combined_context or "").strip()) < 40:
        raise UIValidationError("insufficient writing context to continue safely")


def enforce_style_stability(profile: WritingStyleProfile, generated_text: str) -> dict[str, object]:
    notes: list[str] = []
    changed = False
    lowered = generated_text.casefold()
    if profile.tone == "technical" and "!" in generated_text:
        notes.append("technical tone drift")
        changed = True
    if profile.text_type == WritingTextType.STORY and profile.characters:
        if not any(name.casefold() in lowered for name in profile.characters):
            notes.append("character continuity weak")
    return {"stable": not changed, "notes": notes}


def autonomous_requires_approval(task: WritingTask) -> bool:
    return task.mode.value == "autonomous" and (task.budget.max_words > 500 or task.budget.max_blocks > 4)
