from __future__ import annotations

from jarvis.core.errors import ConfigurationError

from .base import VisionAnalyzer


class VisionAnalyzerRegistry:
    def __init__(self) -> None:
        self._analyzers: dict[str, VisionAnalyzer] = {}

    def register(self, analyzer: VisionAnalyzer) -> None:
        if analyzer.analyzer_name in self._analyzers:
            raise ConfigurationError(f"vision analyzer '{analyzer.analyzer_name}' is already registered")
        self._analyzers[analyzer.analyzer_name] = analyzer

    def get(self, analyzer_name: str) -> VisionAnalyzer | None:
        return self._analyzers.get(analyzer_name)

    def list_analyzers(self) -> list[VisionAnalyzer]:
        return sorted(self._analyzers.values(), key=lambda item: item.analyzer_name)
