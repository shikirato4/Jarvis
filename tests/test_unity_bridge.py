from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "BridgeGame"
    (project / "Assets" / "Scenes").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Assets" / "Scenes" / "Main.unity").write_text("scene")
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


class _BridgeHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        command = body["command"]
        response = {
            "correlation_id": body["correlation_id"],
            "success": True,
            "status": "connected",
            "message": f"handled {command}",
            "warnings": [],
            "data": {"echo": body["payload"], "command": command},
        }
        if command == "ping":
            response["data"]["pong"] = True
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def _start_server() -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), _BridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_unity_bridge_connect_ping_and_disconnect(tmp_path: Path) -> None:
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
    )
    app = build_application(settings)
    app.start()
    try:
        connected = app.runtime_service.unity_connect_bridge({"project": str(project)})
        assert connected.success is True
        health = app.runtime_service.unity_bridge_health(str(project))
        assert health.connected is True
        ping = app.runtime_service.unity_bridge_call({"project": str(project), "command": "ping", "metadata": {"approved": True}})
        assert ping.success is True
        assert ping.data["response"]["data"]["pong"] is True
        disconnected = app.runtime_service.unity_disconnect_bridge({"project": str(project)})
        assert disconnected.success is True
    finally:
        app.stop()
        server.shutdown()
        thread.join(timeout=2)
