from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from fastapi.testclient import TestClient

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "BridgeApiGame"
    (project / "Assets").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


class _BridgeApiHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        response = {
            "correlation_id": body["correlation_id"],
            "success": True,
            "status": "connected",
            "message": "ok",
            "warnings": [],
            "data": {"command": body["command"]},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _start_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), _BridgeApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_unity_bridge_api_routes(monkeypatch, tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    server, thread = _start_server()
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
        unity_bridge_backend_default="http_local",
        unity_bridge_transport_default="http_local",
        unity_bridge_host="127.0.0.1",
        unity_bridge_port=server.server_port,
        unity_require_confirmation_for_bridge_commands=False,
        unity_require_confirmation_for_editor_commands=False,
        unity_require_confirmation_for_launch=False,
    )
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        status = client.get("/unity/bridge/status", params={"project": str(project)})
        assert status.status_code == 200
        connect = client.post("/unity/bridge/connect", json={"project": str(project)})
        assert connect.status_code == 200
        bridge = client.post("/unity/bridge", json={"project": str(project), "command": "ping", "metadata": {"approved": True}})
        assert bridge.status_code == 200
        editor = client.post("/unity/editor/command", json={"project": str(project), "operation_kind": "ping_bridge", "metadata": {"approved": True}})
        assert editor.status_code == 200
    server.shutdown()
    thread.join(timeout=2)
