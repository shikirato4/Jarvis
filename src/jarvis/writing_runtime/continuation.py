from __future__ import annotations

from .generation import WritingGenerator
from .models import WritingAnalysisResult, WritingContinuationRequest, WritingGeneratedBlock


class WritingContinuationEngine:
    def __init__(self, generator: WritingGenerator) -> None:
        self._generator = generator

    def continue_text(self, analysis: WritingAnalysisResult, request: WritingContinuationRequest, *, correlation_id: str) -> WritingGeneratedBlock:
        return self._generator.generate(analysis, prompt=request.instruction or request.prompt, desired_words=request.desired_words, correlation_id=correlation_id)
