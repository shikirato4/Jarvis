from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.system_runtime.backends import NativeApplicationCatalogProvider


def test_system_search_across_multiple_roots(tmp_path: Path) -> None:
    disk_a = tmp_path / "disk_a"
    disk_b = tmp_path / "disk_b"
    (disk_a / "docs").mkdir(parents=True)
    (disk_b / "docs").mkdir(parents=True)
    (disk_a / "docs" / "informe abril.xlsx").write_text("a")
    (disk_b / "docs" / "informe abril final.xlsx").write_text("b")

    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(disk_a, disk_b),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.system_search(
            {
                "resource": {
                    "query": "informe abril",
                    "search_scope": "configured_roots",
                    "max_results": 10,
                }
            }
        )
        assert len(receipt.matches) >= 2
        assert {Path(item.path).anchor for item in receipt.matches if item.path} <= {str(disk_a.anchor), str(disk_b.anchor)}
    finally:
        app.stop()


def test_system_resolve_returns_ambiguous_matches(tmp_path: Path) -> None:
    disk_a = tmp_path / "disk_a"
    disk_b = tmp_path / "disk_b"
    disk_a.mkdir()
    disk_b.mkdir()
    (disk_a / "report.txt").write_text("a")
    (disk_b / "report.txt").write_text("b")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(disk_a, disk_b),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.system_resolve({"query": "report.txt"})
        assert receipt.resolved_target.resolution_status.value == "ambiguous"
        assert len(receipt.resolved_target.ambiguity_candidates) >= 2
    finally:
        app.stop()


def test_native_application_catalog_discovers_desktop_shortcuts_and_apps(tmp_path: Path, monkeypatch) -> None:
    desktop = tmp_path / "Desktop"
    apps = tmp_path / "Apps"
    desktop.mkdir()
    apps.mkdir()
    (desktop / "Opera GX.lnk").write_text("", encoding="utf-8")
    (apps / "WINWORD.EXE").write_text("", encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HOMEDRIVE", str(tmp_path.drive or "C:"))
    monkeypatch.setenv("HOMEPATH", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "EmptyProgramFiles"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "EmptyProgramFilesX86"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))

    provider = NativeApplicationCatalogProvider(tmp_path, {"desktop": str(desktop), "apps": str(apps)})
    applications = provider.list_applications()

    assert any(item.display_name == "Opera GX" for item in applications)
    assert provider.resolve_application("word") is not None
