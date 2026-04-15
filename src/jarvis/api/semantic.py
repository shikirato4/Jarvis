from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Request
from pydantic import Field

from jarvis.core.errors import JarvisError
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.models.base import JarvisBaseModel


class SemanticSearchRequest(JarvisBaseModel):
    query: str
    collection_name: str | None = None
    top_k: int | None = None
    min_score: float | None = None
    source_types: tuple[str, ...] = ()
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    include_summary: bool = True
    correlation_id: str | None = None


class ReindexRequest(JarvisBaseModel):
    collection_name: str | None = None


def install_semantic_routes(app, get_jarvis: Callable[[Request], Any]) -> None:
    @app.get("/semantic/status")
    def semantic_status(request: Request) -> dict[str, Any]:
        return get_jarvis(request).runtime_service.semantic_status()

    @app.get("/semantic/collections")
    def semantic_collections(request: Request) -> dict[str, Any]:
        return {"collections": get_jarvis(request).runtime_service.semantic_collections()}

    @app.post("/semantic/ingest")
    def semantic_ingest(body: DocumentIngestionRequest, request: Request) -> dict[str, Any]:
        try:
            document = get_jarvis(request).runtime_service.semantic_ingest(body)
            return document.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/semantic/search")
    def semantic_search(body: SemanticSearchRequest, request: Request) -> dict[str, Any]:
        try:
            result = get_jarvis(request).runtime_service.semantic_search(
                SemanticSearchQuery(
                    query=body.query,
                    collection_name=body.collection_name,
                    top_k=body.top_k,
                    min_score=body.min_score,
                    source_types=body.source_types,
                    metadata_filters=body.metadata_filters,
                    include_summary=body.include_summary,
                    correlation_id=body.correlation_id,
                )
            )
            return result.model_dump(mode="json")
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc

    @app.post("/semantic/reindex")
    def semantic_reindex(body: ReindexRequest, request: Request) -> dict[str, Any]:
        try:
            return get_jarvis(request).runtime_service.semantic_reindex(body.collection_name)
        except JarvisError as exc:
            raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
