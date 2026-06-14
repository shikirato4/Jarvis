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
    return json.dumps(
        {
            "status": "proposed",
            "summary": "Replace greeting",
            "confidence": 0.8,
            "target_files": [file],
            "operations": [{"type": "replace", "file": file, "old_text": "hello", "new_text": text, "reason": "test"}],
            "warnings": [],
            "tests_suggested": [],
        }
    )


def _set_dual_env(monkeypatch, *, mode: str = "auto", allow_online: bool = False, local: str | None = None, online: str | None = None) -> None:
    monkeypatch.setenv("JARVIS_LLM_MODE", mode)
    monkeypatch.setenv("JARVIS_LOCAL_PROVIDER", "fake")
    monkeypatch.setenv("JARVIS_ONLINE_PROVIDER", "fake-online")
    monkeypatch.setenv("JARVIS_ALLOW_ONLINE_FOR_CODE", "true" if allow_online else "false")
    monkeypatch.setenv("JARVIS_LLM_PREFER_LOCAL", "true")
    if local is None:
        monkeypatch.delenv("JARVIS_LOCAL_LLM_FAKE_RESPONSE", raising=False)
    else:
        monkeypatch.setenv("JARVIS_LOCAL_LLM_FAKE_RESPONSE", local)
    if online is None:
        monkeypatch.delenv("JARVIS_ONLINE_LLM_FAKE_RESPONSE", raising=False)
    else:
        monkeypatch.setenv("JARVIS_ONLINE_LLM_FAKE_RESPONSE", online)
    monkeypatch.delenv("JARVIS_LLM_FAKE_RESPONSE", raising=False)


def test_offline_mode_uses_local_provider(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="offline", local=_response("local"), online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por local en src/app.py", llm_mode="offline")
    result = service.change_propose("cambia hello por local en src/app.py", llm_assisted=True, llm_mode="offline")

    assert route["provider_kind"] == "local"
    assert result["status"] == "proposed"
    assert "+    return 'local'" in result["patch"]["unified_diff"]


def test_online_mode_uses_local_ollama_not_external_llm(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="online", allow_online=True, local=_response("local"), online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por online en src/app.py", llm_mode="online", allow_online=True)
    result = service.change_propose("cambia hello por online en src/app.py", llm_assisted=True, llm_mode="online", allow_online=True)

    assert route["provider_kind"] == "local"
    assert result["status"] == "proposed"
    assert "+    return 'local'" in result["patch"]["unified_diff"]


def test_auto_prefers_local_for_private_code(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="auto", allow_online=True, local=_response("local"), online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por local en src/app.py", llm_mode="auto", allow_online=True)
    result = service.change_propose("cambia hello por local en src/app.py", llm_assisted=True, llm_mode="auto", allow_online=True)

    assert route["provider_kind"] == "local"
    assert "project code" in route["reason"]
    assert "+    return 'local'" in result["patch"]["unified_diff"]


def test_auto_does_not_use_external_online_llm_for_public_task(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="auto", allow_online=True, local=None, online=_response("online", file="README.md"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("agrega nota publica al README", llm_mode="auto", allow_online=True)

    assert route["provider_kind"] == "none"
    assert route["allowed"] is False


def test_secret_context_blocks_online_and_env_never_sent(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="online", allow_online=True, local=_response("local"), online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia TOKEN en .env", llm_mode="online", allow_online=True)
    result = service.change_propose("cambia TOKEN en .env", llm_assisted=True, llm_mode="online", allow_online=True)
    memory = str(service.memory_show()).casefold()

    assert route["allowed"] is False
    assert route["sensitivity"] == "secret"
    assert result["status"] in {"blocked", "needs_review"}
    assert ".env" not in memory
    assert "secret-value" not in memory


def test_offline_local_unavailable_does_not_use_online(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="offline", local=None, online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("cambia hello por hola en src/app.py", llm_mode="offline")
    result = service.change_propose("cambia hello por hola en src/app.py", llm_assisted=True, llm_mode="offline")

    assert route["allowed"] is False
    assert route["provider_kind"] == "local"
    assert "fallback" in str(result).casefold()


def test_auto_local_unavailable_does_not_use_online_llm(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="auto", allow_online=False, local=None, online=_response("online", file="README.md"))
    service = CodeAgentRuntimeService(tmp_path)

    public_route = service.llm_route("explica buenas practicas publicas", llm_mode="auto", allow_online=True)
    private_route = service.llm_route("cambia hello por online en src/app.py", llm_mode="auto")

    assert public_route["provider_kind"] == "none"
    assert public_route["allowed"] is False
    assert private_route["allowed"] is False


def test_auto_online_unavailable_falls_back_to_local(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="auto", allow_online=True, local=_response("local", file="README.md"), online=None)
    service = CodeAgentRuntimeService(tmp_path)

    route = service.llm_route("explica buenas practicas publicas", llm_mode="auto", allow_online=True)

    assert route["provider_kind"] == "local"
    assert route["fallback_used"] is True


def test_disabled_uses_deterministic(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="disabled", local=_response("local"), online=_response("online"))
    service = CodeAgentRuntimeService(tmp_path)

    result = service.change_propose("cambia hello por deterministic en src/app.py", llm_assisted=True, llm_mode="disabled")

    assert result["status"] == "proposed"
    assert "+    return 'deterministic'" in result["patch"]["unified_diff"]
    assert "llm route unavailable" in str(result).casefold()


def test_cli_llm_route_mode_and_offline_change_agent(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="offline", local=_response("local"), online=_response("online"))
    runner = CliRunner()

    route = runner.invoke(app, ["code", "llm", "route", "cambia hello por local en src/app.py", "--root", str(tmp_path), "--mode", "offline"])
    mode = runner.invoke(app, ["code", "llm", "mode", "--root", str(tmp_path)])
    change = runner.invoke(app, ["code", "change", "propose", "cambia hello por local en src/app.py", "--root", str(tmp_path), "--llm-assisted", "--mode", "offline"])
    agent = runner.invoke(app, ["code", "agent", "run", "cambia hello por local en src/app.py", "--root", str(tmp_path), "--dry-run", "--generate-patch", "--llm-assisted", "--mode", "offline"])

    assert route.exit_code == 0
    assert '"provider_kind": "local"' in route.stdout
    assert mode.exit_code == 0
    assert '"mode": "offline"' in mode.stdout
    assert change.exit_code == 0
    assert '"proposed"' in change.stdout
    assert agent.exit_code == 0
    assert '"commands": []' in agent.stdout


def test_router_memory_without_secrets_and_provider_recorded(tmp_path: Path, monkeypatch) -> None:
    _make_project(tmp_path)
    _set_dual_env(monkeypatch, mode="offline", local=_response("local"), online=_response("online"))
    monkeypatch.setenv("JARVIS_LLM_API_KEY", "sk-secret")
    service = CodeAgentRuntimeService(tmp_path)

    service.change_propose("cambia hello por local en src/app.py", llm_assisted=True, llm_mode="offline")
    raw = str(service.memory_show()).casefold()

    assert "llm_events" in raw
    assert "fake-local" in raw
    assert "sk-secret" not in raw
    assert ".env" not in raw
