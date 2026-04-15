from __future__ import annotations

from .base import WritingModelAdapter
from .models import WritingAnalysisResult, WritingGeneratedBlock


class WritingGenerator:
    def __init__(self, models: WritingModelAdapter, settings=None) -> None:
        self._models = models
        self._settings = settings

    def generate(self, analysis: WritingAnalysisResult, *, prompt: str, desired_words: int, correlation_id: str) -> WritingGeneratedBlock:
        response = self._models.infer_json(
            task_type="writing",
            logical_model="general_assistant",
            correlation_id=correlation_id,
            metadata={"component": "writing_runtime", "stage": "generation"},
            timeout_seconds=(getattr(self._settings, "writing_generation_timeout_ms", 90_000) or 90_000) / 1000,
            prompt=(
                "Continue the writing while preserving tone, style and continuity.\n"
                f"User request: {prompt}\n"
                f"Tone: {analysis.style_profile.tone}\n"
                f"Style: {analysis.style_profile.style}\n"
                f"Text type: {analysis.style_profile.text_type.value}\n"
                f"Narrative mode: {analysis.style_profile.narrative_mode}\n"
                f"Point of view: {analysis.style_profile.point_of_view}\n"
                f"Tense: {analysis.style_profile.tense}\n"
                f"Characters: {', '.join(analysis.style_profile.characters)}\n"
                f"Desired words: {desired_words}\n"
                f"Context:\n{analysis.context.combined_context[:3500]}\n"
                'Return JSON with continuation_text, confidence and style_notes.'
            ),
        )
        if response and str(response.get("continuation_text", "")).strip():
            text = str(response["continuation_text"]).strip()
            confidence = float(response.get("confidence") or 0.7)
            notes = [str(item) for item in response.get("style_notes", [])]
            return WritingGeneratedBlock(index=1, text=text, word_count=len(text.split()), confidence=confidence, style_notes=notes)
        return self._fallback(analysis, prompt=prompt, desired_words=desired_words)

    @staticmethod
    def _fallback(analysis: WritingAnalysisResult, *, prompt: str, desired_words: int) -> WritingGeneratedBlock:
        seed = analysis.context.recent_text or analysis.context.visible_text or analysis.context.semantic_context or prompt
        words = seed.split()
        base = " ".join(words[-min(len(words), max(20, desired_words // 2)) :]).strip()
        if analysis.style_profile.text_type.value == "story":
            text = f"{base} La escena continuó con el mismo tono, desarrollando la situación sin romper la coherencia narrativa."
        elif analysis.style_profile.text_type.value == "technical":
            text = f"{base} En continuidad con el texto anterior, el sistema mantiene el mismo nivel técnico y la misma estructura argumentativa."
        else:
            text = f"{base} Continúo con el mismo tono y la misma intención del texto previo."
        return WritingGeneratedBlock(index=1, text=text.strip(), word_count=len(text.split()), confidence=0.55, style_notes=["fallback_generation"])
