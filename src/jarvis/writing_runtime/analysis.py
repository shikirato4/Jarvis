from __future__ import annotations

from .models import WritingAnalysisResult, WritingContext, WritingStyleProfile


class WritingAnalyzer:
    def analyze(self, context: WritingContext, style_profile: WritingStyleProfile) -> WritingAnalysisResult:
        recommendations = []
        if style_profile.text_type.value == "story":
            recommendations.append("preserve narrative continuity and active characters")
        if style_profile.text_type.value == "technical":
            recommendations.append("preserve precision and modular terminology")
        if not context.recent_text:
            recommendations.append("prefer conservative continuation because recent text is weak")
        return WritingAnalysisResult(
            context=context,
            style_profile=style_profile,
            recommendations=recommendations,
            confidence=min((context.source_confidence + style_profile.confidence) / 2, 1.0),
        )
