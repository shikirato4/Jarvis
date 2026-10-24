from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import JarvisError
from jarvis.browser_runtime.base import BrowserSnapshot
from jarvis.browser_runtime.in_memory import InMemoryBrowserBackend
from jarvis.browser_runtime.service import BrowserRuntimeService


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {"_env_file": None, "data_dir": tmp_path / "runtime"}
    values.update(overrides)
    return Settings(**values)


# ---------------------------------------------------------------- backends


def test_in_memory_backend_open_returns_snapshot_with_elements() -> None:
    backend = InMemoryBrowserBackend()

    snapshot = backend.open("https://example.com")

    assert snapshot.url == "https://example.com"
    assert snapshot.title
    assert snapshot.text
    assert any(element.clickable for element in snapshot.elements)
    assert backend.calls == [("open", "https://example.com")]


def test_in_memory_backend_click_and_type_update_state() -> None:
    backend = InMemoryBrowserBackend()
    snapshot = backend.open("https://example.com")
    target = next(e for e in snapshot.elements if e.editable)

    backend.type_text(target.ref, "jarvis browser")
    clicked = backend.click(next(e.ref for e in snapshot.elements if e.clickable))

    assert ("type", target.ref, "jarvis browser") in backend.calls
    assert any(call[0] == "click" for call in backend.calls)
    assert clicked.url == "https://example.com"


def test_in_memory_backend_screenshot_writes_file(tmp_path: Path) -> None:
    backend = InMemoryBrowserBackend()
    backend.open("https://example.com")
    target = tmp_path / "shot.png"

    path = backend.screenshot(target)

    assert path.exists()
    assert backend.calls[-1] == ("screenshot", str(target))


# ---------------------------------------------------------------- service


def _service(tmp_path: Path, **overrides) -> tuple[BrowserRuntimeService, InMemoryBrowserBackend]:
    backend = InMemoryBrowserBackend()
    service = BrowserRuntimeService(_settings(tmp_path, **overrides), backend=backend)
    return service, backend


def test_service_open_returns_snapshot_receipt(tmp_path: Path) -> None:
    service, backend = _service(tmp_path)

    receipt = service.open("https://example.com")

    assert receipt.success is True
    assert isinstance(receipt.snapshot, BrowserSnapshot)
    assert receipt.snapshot.url == "https://example.com"
    assert service.status()["current_url"] == "https://example.com"


def test_service_click_requires_open_page(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    with pytest.raises(JarvisError):
        service.click(0)


def test_service_read_truncates_long_text(tmp_path: Path) -> None:
    service = _service(tmp_path, browser_read_max_chars=50)[0]
    service.open("https://example.com")

    receipt = service.read()

    assert len(receipt.snapshot.text) <= 50


def test_service_press_key_delegates_to_backend(tmp_path: Path) -> None:
    service, backend = _service(tmp_path)
    service.open("https://example.com")

    receipt = service.press_key("Enter")

    assert receipt.success is True
    assert ("press_key", "Enter") in backend.calls


def test_in_memory_backend_press_key_records_call() -> None:
    backend = InMemoryBrowserBackend()
    backend.open("https://example.com")

    backend.press_key("Escape")

    assert ("press_key", "Escape") in backend.calls


# ---------------------------------------------------------- url validation


def test_service_rejects_non_http_schemes(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    for url in ("file:///C:/Windows/system.ini", "ftp://x", "javascript:alert(1)"):
        with pytest.raises(JarvisError):
            service.open(url)


def test_service_allows_localhost_for_routers(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    receipt = service.open("http://192.168.1.1/")

    assert receipt.success is True


# ------------------------------------------------------------ action wiring


def _app(tmp_path: Path, **overrides):
    from jarvis.bootstrap import build_application

    settings = _settings(
        tmp_path,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        voice_tts_provider_default="in_memory",
        voice_tts_provider_fallback_order=("in_memory",),
        **overrides,
    )
    app = build_application(settings)
    return app


def test_browser_actions_registered_with_expected_risk(tmp_path: Path) -> None:
    app = _app(tmp_path)
    try:
        registry = app.action_registry
        expected_risk = {
            "browser.open": "SAFE",
            "browser.read": "SAFE",
            "browser.screenshot": "SAFE",
            "browser.click": "SENSITIVE",
            "browser.type": "SENSITIVE",
            "browser.press_key": "SENSITIVE",
        }
        for name, risk in expected_risk.items():
            definition = registry.get(name)
            assert definition is not None, f"missing action {name}"
            assert definition.risk.value == risk.lower(), f"{name} risk {definition.risk}"
    finally:
        app.stop()


def test_browser_disabled_skips_action_registration(tmp_path: Path) -> None:
    app = _app(tmp_path, browser_enabled=False)
    try:
        assert app.action_registry.get("browser.open") is None
        assert app.browser_runtime_service is None
    finally:
        app.stop()


# ---------------------------------------------------------------- agent tools


def test_browser_tools_registered_for_agent_loop(tmp_path: Path) -> None:
    from jarvis.tools.browser_tools import install_browser_tools
    from jarvis.browser_runtime.in_memory import InMemoryBrowserBackend
    from jarvis.browser_runtime.service import BrowserRuntimeService

    class _FakeToolRegistry:
        def __init__(self) -> None:
            self.definitions = []

        def register(self, definition) -> None:
            self.definitions.append(definition)

    registry = _FakeToolRegistry()
    service = BrowserRuntimeService(
        _settings(tmp_path), backend=InMemoryBrowserBackend()
    )
    install_browser_tools(registry, service)

    names = {tool.name for tool in registry.definitions}
    assert {
        "browser.open",
        "browser.read",
        "browser.click",
        "browser.type",
        "browser.press_key",
        "browser.screenshot",
    } <= names


def test_browser_tool_open_returns_snapshot_data(tmp_path: Path) -> None:
    from jarvis.tools.browser_tools import install_browser_tools
    from jarvis.browser_runtime.in_memory import InMemoryBrowserBackend
    from jarvis.browser_runtime.service import BrowserRuntimeService

    class _FakeToolRegistry:
        def __init__(self) -> None:
            self.definitions = []

        def register(self, definition) -> None:
            self.definitions.append(definition)

    registry = _FakeToolRegistry()
    backend = InMemoryBrowserBackend()
    service = BrowserRuntimeService(_settings(tmp_path), backend=backend)
    install_browser_tools(registry, service)

    open_tool = next(tool for tool in registry.definitions if tool.name == "browser.open")
    payload = open_tool.input_model(url="https://example.com")
    result = open_tool.handler(None, payload)

    assert result.data["url"] == "https://example.com"
    assert result.data["title"] == "Example Domain"
    assert any("[3]" in str(element) or "[0]" in str(element) for element in result.data["elements"])


def test_bootstrap_installs_browser_tools_only_when_enabled(tmp_path: Path) -> None:
    app = _app(tmp_path)
    try:
        names = {tool.name for tool in app.tool_registry.list_tools()}
        assert "browser.open" in names
        assert "browser.press_key" in names
    finally:
        app.stop()

    app2 = _app(tmp_path, browser_enabled=False)
    try:
        names2 = {tool.name for tool in app2.tool_registry.list_tools()}
        assert "browser.open" not in names2
    finally:
        app2.stop()


def test_web_operate_skill_loads_in_catalog(tmp_path: Path) -> None:
    from jarvis.skills_runtime.registry import SkillRegistry

    catalog = SkillRegistry().load_directory(
        Path(__file__).resolve().parent.parent / "skills"
    )
    skills = {skill.name for skill in catalog.skills}
    assert "web-operate" in skills
