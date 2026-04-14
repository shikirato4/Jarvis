from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "LaunchGame"
    (project / "Assets").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


class _Process:
    pid = 4321


def test_unity_launch_project_executes_direct_editor(monkeypatch, tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    editor = tmp_path / "Unity.exe"
    editor.write_text("binary")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
        unity_known_locations={"editor": str(editor)},
        unity_require_confirmation_for_launch=False,
        unity_bridge_enabled=False,
    )
    monkeypatch.setattr("jarvis.unity_runtime.launch.subprocess.Popen", lambda *args, **kwargs: _Process())
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.unity_launch_project({"project": str(project), "metadata": {"approved": True}})
        assert receipt.success is True
        assert receipt.status.value == "launched"
        assert receipt.data["launch"]["editor_pid"] == 4321
    finally:
        app.stop()
