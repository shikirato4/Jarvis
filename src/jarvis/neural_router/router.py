from __future__ import annotations

from .contracts import NeuralRouterBackend, RouterFeatureExtractor
from .models import RouterPrediction, RouterScore, ShadowTrace


class NeuralRouter:
    def __init__(
        self,
        *,
        backend: NeuralRouterBackend,
        feature_extractor: RouterFeatureExtractor,
        tool_map: dict[str, tuple[str, ...]] | None = None,
        agent_intents: set[str] | None = None,
    ) -> None:
        self.backend = backend
        self.feature_extractor = feature_extractor
        self._tool_map = tool_map or {}
        self._agent_intents = agent_intents or {"agent"}

    def predict(self, text: str, *, top_k: int = 3) -> RouterPrediction:
        scores = self.backend.predict_scores(self.feature_extractor.extract(text))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        selected = ranked[0][0]
        candidates: list[str] = []
        for label, _score in ranked[: max(top_k, 1)]:
            for tool in self._tool_map.get(label, ()):
                if tool not in candidates:
                    candidates.append(tool)
        return RouterPrediction(
            intent=selected,
            scores=[RouterScore(label=label, score=score) for label, score in ranked[: max(top_k, 1)]],
            candidate_tools=candidates,
            needs_agent_mode=selected in self._agent_intents,
        )

    def tools_for_intent(self, intent: str) -> list[str]:
        return list(self._tool_map.get(intent, ()))


class ShadowRouter:
    def __init__(self, router: NeuralRouter, *, trace_limit: int = 1_000) -> None:
        self._router = router
        self._trace_limit = max(trace_limit, 1)
        self.traces: list[ShadowTrace] = []

    def predict(self, text: str, *, top_k: int = 3) -> RouterPrediction:
        return self._router.predict(text, top_k=top_k)

    def tools_for_intent(self, intent: str) -> list[str]:
        return self._router.tools_for_intent(intent)

    def observe(
        self,
        *,
        text: str,
        selected_tools: list[str],
        success: bool,
        step_count: int,
        latency_ms: float,
        correction_required: bool,
    ) -> list[str]:
        prediction = self._router.predict(text)
        self.traces.append(
            ShadowTrace(
                intent_class=prediction.intent,
                candidate_tools=prediction.candidate_tools,
                selected_tools=list(selected_tools),
                success=success,
                step_count=step_count,
                latency_ms=latency_ms,
                correction_required=correction_required,
            )
        )
        if len(self.traces) > self._trace_limit:
            del self.traces[: len(self.traces) - self._trace_limit]
        return list(selected_tools)

