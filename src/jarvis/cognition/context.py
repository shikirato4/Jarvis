from __future__ import annotations

from jarvis.memory_semantic.base import RetrievedContext


class RetrievedContextFormatter:
    def __init__(self, *, char_budget: int) -> None:
        self._char_budget = char_budget

    def format_for_prompt(self, context: RetrievedContext) -> str:
        if not context.chunks:
            return ""
        consumed = 0
        lines = ["Recovered context:"]
        for chunk in context.chunks:
            source = chunk.source_path or chunk.collection_name
            line = f"- [{chunk.rank}] ({chunk.score:.3f}) {source}: {chunk.content.strip()}"
            if consumed + len(line) > self._char_budget:
                break
            lines.append(line)
            consumed += len(line)
        return "\n".join(lines)

    def findings(self, context: RetrievedContext) -> list[str]:
        return [chunk.content.strip() for chunk in context.chunks]
