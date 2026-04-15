from __future__ import annotations

from pathlib import Path

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.system_runtime.backends import NativeApplicationCatalogProvider


def _build_system_app(tmp_path: Path):
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )
    app = build_application(settings)
    app.start()
    return app


def test_system_runtime_open_path_returns_receipt(tmp_path: Path, monkeypatch) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("jarvis")
    app = _build_system_app(tmp_path)
    try:
        monkeypatch.setattr(app.ui_automation_service, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ui automation should not be used")))  # noqa: ARG005
        receipt = app.runtime_service.system_open_path(str(document), dry_run=True)
        assert receipt.status.value == "opened"
        assert receipt.resolved_target is not None
        assert receipt.resolved_target.path == str(document)
    finally:
        app.stop()


def test_system_runtime_open_requires_confirmation_for_sensitive_target(tmp_path: Path) -> None:
    sensitive = tmp_path / "system-root"
    sensitive.mkdir()
    document = sensitive / "config.txt"
    document.write_text("jarvis")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
        system_sensitive_roots=(sensitive,),
    )
    app = build_application(settings)
    app.start()
    try:
        receipt = app.runtime_service.system_open_path(str(document), dry_run=True)
        assert receipt.status.value == "confirmation_required"
        assert receipt.confirmation_required is True
    finally:
        app.stop()


def test_system_runtime_open_trusted_applications_without_confirmation(tmp_path: Path) -> None:
    app = _build_system_app(tmp_path)
    try:
        for target in ("word", "notepad", "vscode", "chrome", "opera", "calculadora", "explorador"):
            receipt = app.runtime_service.system_open_application(target, dry_run=True)
            assert receipt.confirmation_required is False
            assert receipt.success is True
            assert receipt.status.value == "launched"
    finally:
        app.stop()


def test_system_runtime_open_unknown_application_returns_clear_resolution_error(tmp_path: Path) -> None:
    app = _build_system_app(tmp_path)
    try:
        try:
            app.runtime_service.system_open_application("jarvis-no-existe", dry_run=True)
        except Exception as exc:  # noqa: BLE001
            assert "not_found" in str(exc) or "not resolved" in str(exc)
        else:
            raise AssertionError("expected resolution failure")
    finally:
        app.stop()


def test_native_application_catalog_provider_uses_registry_aliases(tmp_path: Path, monkeypatch) -> None:
    provider = NativeApplicationCatalogProvider(workspace_root=tmp_path)
    fake_word = tmp_path / "WINWORD.EXE"
    fake_word.write_text("", encoding="utf-8")
    fake_code = tmp_path / "Code.exe"
    fake_code.write_text("", encoding="utf-8")

    def _fake_app_path(executable_name: str):
        if executable_name.casefold() == "winword.exe":
            return fake_word
        if executable_name.casefold() == "code.exe":
            return fake_code
        return None

    monkeypatch.setattr("jarvis.system_runtime.backends._resolve_windows_app_path", _fake_app_path)

    word = provider.resolve_application("microsoft word")
    code = provider.resolve_application("visual studio code")

    assert word is not None
    assert word.display_name == "Microsoft Word"
    assert word.path == str(fake_word)
    assert code is not None
    assert code.display_name == "Visual Studio Code"
    assert code.path == str(fake_code)
