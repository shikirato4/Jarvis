from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.config import Settings
from jarvis.tools.registry import ToolRegistry
from jarvis.tools.web_tools import WebFetchPageToolPayload, install_web_tools
from jarvis.web_search.fetcher import RawPage, SourceFetcher

_URL = "https://example.com/articulo"
_PUBLIC_DNS = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


def _fetcher_with(body: str = "Contenido principal del articulo.", *, content_type: str = "text/html") -> SourceFetcher:
    page = RawPage(
        url=_URL,
        content_type=content_type,
        body=body.encode("utf-8"),
        headers={"Content-Type": content_type},
    )
    return SourceFetcher(
        resolver=lambda *_args, **_kwargs: _PUBLIC_DNS,
        transport=lambda url, timeout, max_bytes: page,
    )


def _registry() -> ToolRegistry:
    return ToolRegistry(
        settings=None,
        memory=None,
        action_router=None,
        models=None,
        event_bus=None,
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        action_router=None,
        correlation_id="corr-1",
        dry_run=False,
        metadata={},
    )


def test_fetch_page_tool_returns_title_window_and_links() -> None:
    html = (
        "<html><head><title>El articulo</title></head>"
        "<body><h1>El articulo</h1><p>Contenido principal del articulo.</p>"
        "<a href='https://example.com/otro'>Otro</a></body></html>"
    )
    registry = _registry()
    install_web_tools(registry, _fetcher_with(html))

    tool = registry.get("web.fetch_page")
    assert tool is not None
    result = tool.handler(_context(), WebFetchPageToolPayload(url=_URL))

    assert result.status.value == "success"
    assert result.data["title"] == "El articulo"
    assert "Contenido principal" in result.data["text"]
    assert result.data["final_url"] == _URL
    assert "example.com" in result.data["domain"]
    assert "https://example.com/otro" in result.data["links"]
    assert result.data["total_chars"] == len(result.data["text"]) or result.data["total_chars"] > 0
    assert result.data["has_more"] is False


def test_fetch_page_tool_paginates_long_content() -> None:
    long_text = "x" * 9_000
    registry = _registry()
    install_web_tools(registry, _fetcher_with(long_text))

    tool = registry.get("web.fetch_page")
    result = tool.handler(
        _context(),
        WebFetchPageToolPayload(url=_URL, offset_chars=0, max_chars=4000),
    )

    assert result.status.value == "success"
    assert result.data["total_chars"] == 9_000
    assert len(result.data["text"]) == 4_000
    assert result.data["has_more"] is True

    second = tool.handler(
        _context(),
        WebFetchPageToolPayload(url=_URL, offset_chars=8000, max_chars=4000),
    )
    assert len(second.data["text"]) == 1_000
    assert second.data["has_more"] is False


def test_fetch_page_tool_blocks_unsafe_urls() -> None:
    registry = _registry()
    install_web_tools(registry, _fetcher_with())

    tool = registry.get("web.fetch_page")
    assert tool is not None
    result = tool.handler(_context(), WebFetchPageToolPayload(url="http://127.0.0.1:8000/admin"))
    assert result.status.value == "failed"
    assert "blocked" in result.message or "blocked" in str(result.data.get("status", ""))


def test_fetch_page_tool_rejects_invalid_payloads() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WebFetchPageToolPayload(url=_URL, max_chars=10)
    with pytest.raises(ValidationError):
        WebFetchPageToolPayload(url=_URL, offset_chars=-1)


def test_install_web_tools_registers_only_the_fetch_page_tool() -> None:
    registry = _registry()

    install_web_tools(registry, _fetcher_with())

    names = [tool.name for tool in registry.list_tools()]
    assert names == ["web.fetch_page"]


def test_bootstrap_wires_web_fetch_tool(tmp_path: Path) -> None:
    from jarvis.bootstrap import build_application

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
    )

    app = build_application(settings)
    try:
        tool = app.tool_registry.get("web.fetch_page")
        assert tool is not None
        assert "web" in tool.tags
    finally:
        app.stop()
