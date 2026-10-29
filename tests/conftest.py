from pathlib import Path
import ipaddress
import socket

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _is_loopback_host(host: object) -> bool:
    value = host.decode() if isinstance(host, bytes) else str(host)
    if value.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_unmarked_external_network(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Fail unit tests that accidentally contact non-loopback services."""
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("e2e"):
        return

    # A developer's real .env must never turn a unit test into a live provider test.
    for name in (
        "BRAVE_SEARCH_API_KEY",
        "JARVIS_BRAVE_SEARCH_API_KEY",
        "YDC_API_KEY",
        "JARVIS_YDC_API_KEY",
        "EXA_API_KEY",
        "JARVIS_EXA_API_KEY",
        "NVIDIA_API_KEY",
        "JARVIS_NVIDIA_API_KEY",
    ):
        monkeypatch.setenv(name, "")

    # Nor must it stretch the request budgets that streaming tests pin down.
    for name in (
        "JARVIS_LOCAL_LLM_TIMEOUT_SECONDS",
        "LOCAL_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback_host(host):
            raise AssertionError(
                f"External network access is forbidden in unit tests: {host!s}. "
                "Use a fake/MockTransport or mark an opt-in integration/e2e test."
            )
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback_host(host):
            raise AssertionError(
                f"External network access is forbidden in unit tests: {host!s}. "
                "Use a fake/MockTransport or mark an opt-in integration/e2e test."
            )
        return original_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def isolate_optional_audio_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent from local audio hardware and user .env choices."""
    monkeypatch.setenv("JARVIS_VOICE_AUDIO_INPUT_BACKEND_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_AUDIO_OUTPUT_BACKEND_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_TTS_PROVIDER_DEFAULT", "in_memory")
    monkeypatch.setenv("JARVIS_VOICE_TTS_PROVIDER_FALLBACK_ORDER", '["in_memory"]')


@pytest.fixture()
def jarvis_app(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        command_allowlist=("where",),
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        local_runtime_autostart=False,
        nvidia_api_key=None,
        web_search_enabled=False,
    )
    app = build_application(settings)
    # Research unit tests exercise deterministic fallbacks, never installed or
    # credential-backed model providers that happen to exist on the host.
    app.research_runtime_service._pipeline._analyzer._models.infer_json = (  # noqa: SLF001
        lambda **_kwargs: None
    )
    app.start()
    try:
        yield app
    finally:
        app.stop()
