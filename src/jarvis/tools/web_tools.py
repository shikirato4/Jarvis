from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from jarvis.tools.models import ToolExecutionStatus, ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry
from jarvis.web_search.fetcher import SourceFetcher
from jarvis.web_search.models import SourceCandidate


class WebFetchPageToolPayload(BaseModel):
    """Fetch a public page and page through its extracted main content."""

    url: str
    offset_chars: int = Field(default=0, ge=0)
    max_chars: int = Field(default=4_000, ge=500, le=8_000)
    max_links: int = Field(default=20, ge=0, le=50)


def install_web_tools(registry: ToolRegistry, fetcher: SourceFetcher) -> None:
    """Register the read-only web browsing tools for the agent loop."""
    registry.register(
        ToolDefinition(
            name="web.fetch_page",
            description=(
                "Fetch a public web page and return its title, a text window of "
                "its main content, and up to 20 of its links. Use offset_chars "
                "and max_chars to page through long content. Unsafe URLs "
                "(private networks, non-http schemes) are blocked."
            ),
            input_model=WebFetchPageToolPayload,
            handler=_make_fetch_page_handler(fetcher),
            tags=("web", "research"),
        )
    )


def _make_fetch_page_handler(fetcher: SourceFetcher):
    def handler(context: ToolContext, payload: WebFetchPageToolPayload) -> ToolResult:
        candidate = SourceCandidate(
            source_id=f"agent-{uuid4().hex[:12]}",
            title="",
            url=payload.url,
            normalized_url=payload.url,
            domain="",
        )
        fetched = fetcher.fetch(candidate)
        if fetched.status != "ok" or fetched.error:
            return ToolResult(
                status=ToolExecutionStatus.FAILED,
                message=(
                    f"web.fetch_page failed: {fetched.status}"
                    f"{': ' + fetched.error if fetched.error else ''}"
                ),
                data={
                    "url": payload.url,
                    "status": fetched.status,
                    "error": fetched.error or "",
                },
            )
        text = fetched.text
        window = text[payload.offset_chars : payload.offset_chars + payload.max_chars]
        end = payload.offset_chars + len(window)
        return ToolResult(
            message=f"fetched '{fetched.title or payload.url}'",
            data={
                "url": payload.url,
                "final_url": fetched.final_url,
                "title": fetched.title,
                "domain": fetched.domain,
                "content_type": fetched.content_type,
                "text": window,
                "offset_chars": payload.offset_chars,
                "returned_chars": len(window),
                "total_chars": len(text),
                "has_more": end < len(text),
                "links": fetched.links[: payload.max_links],
            },
        )

    return handler
