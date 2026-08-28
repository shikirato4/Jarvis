from __future__ import annotations

from jarvis.memory_semantic.base import RetrievedContext

from .models import ContextFragment, ContextSource


def context_fragments_from_retrieved_context(
    context: RetrievedContext | None,
    *,
    limit: int | None = None,
) -> list[ContextFragment]:
    """Adapt semantic results without copying paths or private provenance."""

    if context is None:
        return []
    ranked = sorted(context.chunks, key=lambda chunk: (-chunk.score, chunk.rank))
    if limit is not None:
        ranked = ranked[: max(int(limit), 0)]
    return [
        ContextFragment(
            fragment_id=chunk.chunk_id,
            source=ContextSource.MEMORY,
            content=chunk.content,
            score=chunk.score,
            metadata={
                "source_type": chunk.source_type,
                "rank": chunk.rank,
            },
        )
        for chunk in ranked
    ]
