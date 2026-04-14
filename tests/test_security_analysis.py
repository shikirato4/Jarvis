from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings
from jarvis.core.errors import SafetyViolationError
from jarvis.security_runtime import SecurityAnalyzeRequest, SecurityPasswordCheckRequest


def test_security_analyze_detects_python_issues(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    jarvis.start()
    try:
        result = jarvis.runtime_service.security_analyze(
            SecurityAnalyzeRequest(
                code="""
import subprocess
password = "secret123"
eval("print(1)")
subprocess.run("dir", shell=True)
""",
            )
        )
        rule_ids = {item.rule_id for item in result.findings}
        assert {"PY001", "PY002", "PY009"} <= rule_ids

        password_review = jarvis.runtime_service.security_check_password(SecurityPasswordCheckRequest(password="Password123"))
        assert password_review.metadata["tier"] == "weak"
    finally:
        jarvis.stop()


def test_security_blocks_dangerous_requests(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    jarvis.start()
    try:
        with pytest.raises(SafetyViolationError):
            jarvis.runtime_service.security_analyze(SecurityAnalyzeRequest(query="quiero hackear cuentas reales"))
    finally:
        jarvis.stop()


def test_security_api_routes(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        analyze = client.post("/security/analyze", json={"query": "OWASP Top 10"})
        assert analyze.status_code == 200
        assert "summary" in analyze.json()
        password = client.post("/security/check-password", json={"password": "S3gura!Passphrase2026"})
        assert password.status_code == 200
        assert password.json()["metadata"]["tier"] in {"moderate", "strong"}


def test_security_scans_secrets_and_dependencies(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text('api_key = "ABCDEFGHIJKLMNOP"\n')
    (project / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"
dependencies = ["PyYAML>=6.0", "Flask<2.0"]
""".strip(),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=project,
        research_allowed_roots=(project,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    jarvis.start()
    try:
        secrets = jarvis.runtime_service.security_analyze(SecurityAnalyzeRequest(audit_kind="secrets", path=str(project)))
        assert any(item.rule_id.startswith("SECRET") for item in secrets.findings)

        deps = jarvis.runtime_service.security_analyze(SecurityAnalyzeRequest(audit_kind="dependencies", path=str(project)))
        assert {item.rule_id for item in deps.findings} >= {"DEP001", "DEP002"}
    finally:
        jarvis.stop()


def test_security_local_port_scan_restricted_to_localhost(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    jarvis = build_application(settings)
    jarvis.start()
    try:
        result = jarvis.runtime_service.security_analyze(
            SecurityAnalyzeRequest(audit_kind="local_ports", host="127.0.0.1", ports=[1, 11434], timeout_ms=100)
        )
        assert result.category == "local_ports"
        with pytest.raises(SafetyViolationError):
            jarvis.runtime_service.security_analyze(
                SecurityAnalyzeRequest(audit_kind="local_ports", host="8.8.8.8", ports=[53], timeout_ms=100)
            )
    finally:
        jarvis.stop()


def test_security_cli_commands(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
    )
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(settings))

    password = runner.invoke(app, ["security", "check-password", "hunter2"])
    analyze = runner.invoke(app, ["security", "analyze", "--query", "OWASP Top 10"])

    assert password.exit_code == 0
    assert analyze.exit_code == 0
