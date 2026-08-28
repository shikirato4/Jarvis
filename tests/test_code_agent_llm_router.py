from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService


def _make_project(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret-value\n", encoding="utf-8")


def _response(text: str = "hola", file: str = "src/app.py") -> str:
    return json.dumps({
        "status": "proposed",
        "summary": "Replace greeting",
        "confidence": 0.8,
        "target_files": [file],
        "operations": [{
            "type": "replace", "file": file, "old_text": "hello",
            "new_text": text, "reason": "test",
        }],
        "warnings": [],
        "tests_suggested": [],
    })


def _set_dual_env(monkeypatch, *, mode: str = "auto", local: str | None = None, remote: str | None = None) -> None:
    monkeypatch.setenv("JARVIS_LLM_MODE", mode)
    monkeypatch.setenv("JARVIS_LOCAL_PROVIDER", "fake-local")
    monkeypatch.setenv("JARVIS_ONLINE_PROVIDER", "fake-remote")
    if local is None:
        monkeypatch.delenv("JARVIS_LOCAL_LLM_FAKE_RESPONSE", raising=False)
    else:
        monkeypatch.setenv("JARVIS_LOCAL_LLM_FAKE_RESPONSE", local)
    if remote is None:
        monkeypatch.delenv("JARVIS_ONLINE_LLM_FAKE_RESPONSE", raising=False)
    else:
        monkeypatch.setenv("JARVIS_ONLINE_LLM_FAKE_RESPONSE", remote)
    monkeypatch.delenv("JARVIS_LLM_FAKE_RESPONSE", raising=False)


def test_local_mode_uses_only_local_provider(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="local", local=_response("local"), remote=_response("remote"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por local en src/app.py", llm_mode="local")
    result = service.change_propose(
        "cambia hello por local en src/app.py", llm_assisted=True, llm_mode="local"
    )

    assert route["provider_kind"] == "local"
    assert route["provider_name"] == "fake-local"
    assert result["status"] == "proposed"
    assert "+    return 'local'" in result["patch"]["unified_diff"]


def test_nvidia_api_mode_uses_only_remote_provider(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="nvidia_api", local=_response("local"), remote=_response("remote"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por remote en src/app.py", llm_mode="nvidia_api")
    result = service.change_propose(
        "cambia hello por remote en src/app.py", llm_assisted=True, llm_mode="nvidia_api"
    )

    assert route["provider_kind"] == "remote"
    assert route["provider_name"] == "fake-online"
    assert "+    return 'remote'" in result["patch"]["unified_diff"]


def test_auto_prefers_remote_then_falls_back_to_local(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, local=_response("local"), remote=_response("remote"))
    preferred_service = CodeAgentRuntimeService(tmp_path)
    preferred = preferred_service.llm_route("cambia hello por remote en src/app.py", llm_mode="auto")

    assert preferred["provider_kind"] == "remote"
    assert preferred["fallback_used"] is False

    _set_dual_env(monkeypatch, local=_response("local"), remote=None)
    fallback_service = CodeAgentRuntimeService(tmp_path)
    fallback = fallback_service.llm_route("cambia hello por local en src/app.py", llm_mode="auto")

    assert fallback["provider_kind"] == "local"
    assert fallback["fallback_used"] is True


def test_strict_modes_never_cross_fallback_boundary(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="local", local=None, remote=_response("remote"))
    local_route = CodeAgentRuntimeService(tmp_path).llm_route("actualiza src/app.py", llm_mode="local")

    _set_dual_env(monkeypatch, mode="nvidia_api", local=_response("local"), remote=None)
    remote_route = CodeAgentRuntimeService(tmp_path).llm_route(
        "actualiza src/app.py", llm_mode="nvidia_api"
    )

    assert local_route["provider_kind"] == "local"
    assert local_route["allowed"] is False
    assert remote_route["provider_kind"] == "remote"
    assert remote_route["allowed"] is False


def test_secret_context_blocks_every_provider_and_is_redacted(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="nvidia_api", local=_response("local"), remote=_response("remote"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia TOKEN en .env", llm_mode="nvidia_api")
    result = service.change_propose("cambia TOKEN en .env", llm_assisted=True, llm_mode="nvidia_api")
    memory = str(service.memory_show()).casefold()

    assert route["allowed"] is False
    assert route["sensitivity"] == "secret"
    assert result["status"] in {"blocked", "needs_review"}
    assert ".env" not in memory
    assert "secret-value" not in memory


def test_invalid_mode_is_rejected_as_migrated_auto(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="deprecated", local=_response("local"), remote=_response("remote"))
    route = CodeAgentRuntimeService(tmp_path).llm_route("actualiza src/app.py", llm_mode="deprecated")

    assert route["mode"] == "auto"
    assert route["allowed"] is False
    assert route["provider_kind"] == "none"


def test_cli_route_and_change_accept_exact_local_mode(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="local", local=_response("local"))
    runner = CliRunner()

    route = runner.invoke(app, [
        "code", "llm", "route", "cambia hello por local en src/app.py",
        "--root", str(tmp_path), "--mode", "local",
    ])
    mode = runner.invoke(app, ["code", "llm", "mode", "--root", str(tmp_path)])
    change = runner.invoke(app, [
        "code", "change", "propose", "cambia hello por local en src/app.py",
        "--root", str(tmp_path), "--llm-assisted", "--mode", "local",
    ])

    assert route.exit_code == 0
    assert '"provider_kind": "local"' in route.stdout
    assert mode.exit_code == 0
    assert '"mode": "local"' in mode.stdout
    assert change.exit_code == 0
    assert '"proposed"' in change.stdout


def test_router_memory_records_provider_without_secrets(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="local", local=_response("local"))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-secret")
    service = CodeAgentRuntimeService(tmp_path)

    service.change_propose(
        "cambia hello por local en src/app.py", llm_assisted=True, llm_mode="local"
    )
    raw = str(service.memory_show()).casefold()

    assert "llm_events" in raw
    assert "fake-local" in raw
    assert "nvapi-secret" not in raw
    assert ".env" not in raw
