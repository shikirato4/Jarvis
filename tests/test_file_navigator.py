from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis.file_runtime import FileErrorCode, FileNavigator, FileOperationError


@pytest.fixture
def navigator(tmp_path: Path) -> FileNavigator:
    return FileNavigator(tmp_path, max_search_results=10, max_search_depth=3)


def test_resolve_absolute_relative_workspace_and_unicode(navigator: FileNavigator, tmp_path: Path) -> None:
    absolute = tmp_path / "área con espacios"
    assert navigator.resolve_path(absolute) == absolute.resolve()
    assert navigator.resolve_path("área con espacios") == absolute.resolve()
    assert navigator.resolve_path("workspace") == tmp_path.resolve()


def test_windows_known_folders_are_absolute(navigator: FileNavigator) -> None:
    folders = navigator.known_folders()
    assert {"desktop", "documents", "downloads", "pictures", "videos", "music", "workspace"} <= folders.keys()
    assert all(Path(value).is_absolute() for value in folders.values())


def test_list_directory_metadata_and_pagination(navigator: FileNavigator, tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("bb", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    page = navigator.list_directory(limit=2)
    assert [item["name"] for item in page["items"]] == ["a.txt", "b.txt"]
    assert page["truncated"] is True
    assert page["total"] == 3
    assert page["items"][0]["extension"] == ".txt"


def test_search_filename_extension_depth_limit_and_symlink_safety(navigator: FileNavigator, tmp_path: Path) -> None:
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "report.pdf").write_bytes(b"%PDF")
    (nested / "report.txt").write_text("report", encoding="utf-8")
    try:
        os.symlink(tmp_path, nested / "loop", target_is_directory=True)
    except OSError:
        pass
    result = navigator.search("report", extension="pdf", max_depth=3)
    assert [item["name"] for item in result["items"]] == ["report.pdf"]
    assert result["scanned"] < 20


def test_search_limit_and_cancellation(navigator: FileNavigator, tmp_path: Path) -> None:
    for index in range(8):
        (tmp_path / f"item-{index}.txt").write_text("x", encoding="utf-8")
    result = navigator.search("*.txt", max_results=2)
    assert len(result["items"]) == 2
    assert result["truncated"] is True
    with pytest.raises(FileOperationError) as error:
        navigator.search("*", cancel_check=lambda: True)
    assert error.value.code is FileErrorCode.CANCELLED


def test_stat_file_directory_type_and_hash(navigator: FileNavigator, tmp_path: Path) -> None:
    file_path = tmp_path / "foto.jpg.exe"
    file_path.write_bytes(b"hello")
    file_stat = navigator.stat(file_path, sha256=True)["metadata"]
    directory_stat = navigator.stat(tmp_path)["metadata"]
    assert file_stat["extension"] == ".exe"
    assert file_stat["mime_type"] != "image/jpeg"
    assert len(file_stat["sha256"]) == 64
    assert directory_stat["is_directory"] is True


def test_create_directory_file_utf8_and_conflict(navigator: FileNavigator, tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "pruebas"
    assert navigator.create_directory(directory, parents=True)["verified"] is True
    file_path = directory / "saludo.txt"
    assert navigator.create_file(file_path, content="¡Hola! 漢字")["verified"] is True
    assert file_path.read_text(encoding="utf-8") == "¡Hola! 漢字"
    with pytest.raises(FileOperationError) as error:
        navigator.create_file(file_path, content="overwrite")
    assert error.value.code is FileErrorCode.ALREADY_EXISTS
    assert file_path.read_text(encoding="utf-8") == "¡Hola! 漢字"


def test_explicit_replace_and_rename_new(navigator: FileNavigator, tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("old", encoding="utf-8")
    navigator.create_file(target, content="new", overwrite="replace")
    renamed = navigator.create_file(target, content="other", overwrite="rename_new")
    assert target.read_text(encoding="utf-8") == "new"
    assert Path(renamed["source"]).name == "a (1).txt"


def test_copy_move_and_rename_files_and_directories(navigator: FileNavigator, tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("abc", encoding="utf-8")
    copied = tmp_path / "copied.txt"
    assert navigator.copy(source, copied)["verified"] is True
    moved = tmp_path / "moved.txt"
    assert navigator.move(copied, moved)["verified"] is True
    renamed = navigator.rename(moved, "renamed.txt")
    assert renamed["verified"] is True
    assert not moved.exists() and (tmp_path / "renamed.txt").exists()
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "inside.txt").write_text("x", encoding="utf-8")
    assert navigator.copy(folder, tmp_path / "folder-copy")["verified"] is True
    assert navigator.rename(folder, "folder-renamed")["verified"] is True


def test_rename_rejects_traversal(navigator: FileNavigator, tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("x", encoding="utf-8")
    with pytest.raises(FileOperationError) as error:
        navigator.rename(source, "../escape.txt")
    assert error.value.code is FileErrorCode.INVALID_NAME


def test_batch_move_reports_partial_failure(navigator: FileNavigator, tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("x", encoding="utf-8")
    destination = tmp_path / "dest"
    destination.mkdir()
    result = navigator.batch_move([source, tmp_path / "missing.txt"], destination)
    assert result["status"] == "partial_failure"
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["error_code"] == "NOT_FOUND"


def test_read_text_is_bounded_and_rejects_binary(navigator: FileNavigator, tmp_path: Path) -> None:
    text_path = tmp_path / "large.log"
    text_path.write_text("abcdef", encoding="utf-8")
    result = navigator.read_text(text_path, max_bytes=3)
    assert result["content"] == "abc"
    assert result["truncated"] is True
    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"a\x00b")
    with pytest.raises(FileOperationError):
        navigator.read_text(binary)


def test_permanent_delete_disposable_fixture(navigator: FileNavigator, tmp_path: Path) -> None:
    disposable = tmp_path / "disposable.txt"
    disposable.write_text("delete me", encoding="utf-8")
    result = navigator.delete(disposable, permanent=True)
    assert result["verified"] is True
    assert not disposable.exists()
