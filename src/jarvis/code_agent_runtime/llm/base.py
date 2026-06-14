from __future__ import annotations

from abc import ABC, abstractmethod

from jarvis.code_agent_runtime.llm.models import LLMGenerateRequest, LLMGenerateResult


class LLMProvider(ABC):
    provider_name: str = "base"
    model_name: str = "unknown"

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate_change_proposal(self, request: LLMGenerateRequest) -> LLMGenerateResult:
        raise NotImplementedError
