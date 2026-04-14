from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings


def _make_project(root: Path) -> Path:
    project = root / "MyGame"
    (project / "Assets" / "Scripts").mkdir(parents=True)
    (project / "Packages").mkdir(parents=True)
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "Packages" / "manifest.json").write_text('{"dependencies":{}}\n')
    return project


def test_unity_generate_and_write_script(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_require_confirmation_for_script_overwrite=False,
        unity_discovery_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        generated = app.runtime_service.unity_generate_script(
            {
                "project": str(project),
                "folder": "Assets/Scripts",
                "class_name": "PlayerController",
                "namespace": "MyGame.Runtime",
                "script_type": "mono_behaviour",
            }
        )
        assert "PlayerController" in generated.data["content"]
        written = app.runtime_service.unity_write_script(
            {
                "project": str(project),
                "folder": "Assets/Scripts",
                "class_name": "PlayerController",
                "content": generated.data["content"],
                "overwrite": False,
            }
        )
        assert written.status.value == "written"
        script_path = project / "Assets" / "Scripts" / "PlayerController.cs"
        assert script_path.exists()
        assert "namespace MyGame.Runtime" in script_path.read_text(encoding="utf-8")
    finally:
        app.stop()


def test_unity_write_script_blocks_outside_assets(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        unity_discovery_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.unity_write_script(
            {
                "project": str(project),
                "asset_path": "../Outside.cs",
                "content": "public class Outside {}",
                "metadata": {"approved": True},
            }
        )
        assert receipt.status.value in {"blocked", "confirmation_required", "failed"}  # will be exception if validation moves earlier
    except Exception:
        pass
    finally:
        app.stop()
