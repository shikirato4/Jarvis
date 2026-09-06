from __future__ import annotations

from uuid import uuid4

from .base import WritingModelAdapter
from .models import WritingContext, WritingStyleProfile, WritingTextType
from .parsing import normalize_confidence


class WritingStyleAnalyzer:
    def __init__(self, models: WritingModelAdapter) -> None:
        self._models = models

    def analyze(self, context: WritingContext, *, correlation_id: str) -> WritingStyleProfile:
        prompt = (
            "Analyze writing style.\n"
            f"Context:\n{context.combined_context[:3000]}\n"
            'Return JSON with tone, style, text_type, narrative_mode, point_of_view, tense, paragraph_structure, characters, confidence.'
        )
        response = self._models.infer_json(
            task_type="analysis",
            logical_model="general_assistant",
            prompt=prompt,
            correlation_id=correlation_id,
            metadata={"component": "writing_runtime", "stage": "style_analysis"},
        )
        if response:
            text_type = str(response.get("text_type") or "unknown")
            try:
                parsed_type = WritingTextType(text_type)
            except Exception:
                parsed_type = WritingTextType.UNKNOWN
            if parsed_type is WritingTextType.UNKNOWN:
                parsed_type = _infer_text_type(context.combined_context)
            raw_characters = response.get("characters")
            characters = (
                [str(item) for item in raw_characters]
                if isinstance(raw_characters, (list, tuple))
                else []
            )
            return WritingStyleProfile(
                profile_id=str(uuid4()),
                language=context.language,
                tone=str(response.get("tone") or "neutral"),
                style=str(response.get("style") or "balanced"),
                text_type=parsed_type,
                narrative_mode=str(response.get("narrative_mode") or "unknown"),
                point_of_view=response.get("point_of_view"),
                tense=response.get("tense"),
                paragraph_structure=response.get("paragraph_structure"),
                characters=characters,
                confidence=normalize_confidence(response.get("confidence"), default=0.6),
            )
        return self._fallback(context)

    def _fallback(self, context: WritingContext) -> WritingStyleProfile:
        text = context.combined_context.casefold()
        text_type = WritingTextType.STORY if any(token in text for token in ("capítulo", "escena", "dijo")) else WritingTextType.TECHNICAL if any(token in text for token in ("sistema", "arquitectura", "módulo")) else WritingTextType.CASUAL
        tone = "narrative" if text_type == WritingTextType.STORY else "technical" if text_type == WritingTextType.TECHNICAL else "neutral"
        return WritingStyleProfile(
            profile_id=str(uuid4()),
            language=context.language,
            tone=tone,
            style="balanced",
            text_type=text_type,
            narrative_mode="scene" if text_type == WritingTextType.STORY else "expository",
            point_of_view="third_person" if text_type == WritingTextType.STORY else None,
            tense="present",
            paragraph_structure="medium",
            confidence=0.55,
            characters=_extract_capitalized_names(context.combined_context),
        )


def _extract_capitalized_names(text: str) -> list[str]:
    names: list[str] = []
    for token in text.replace("\n", " ").split():
        cleaned = token.strip(".,;:!?\"'()[]{}")
        if len(cleaned) > 2 and cleaned[:1].isupper():
            if cleaned not in names:
                names.append(cleaned)
    return names[:8]


def _infer_text_type(text: str) -> WritingTextType:
    lowered = text.casefold()
    if any(token in lowered for token in ("capítulo", "escena", "personaje", "dijo", "pasillo", "respiración")):
        return WritingTextType.STORY
    if any(token in lowered for token in ("sistema", "arquitectura", "módulo", "implementación", "función")):
        return WritingTextType.TECHNICAL
    return WritingTextType.CASUAL
