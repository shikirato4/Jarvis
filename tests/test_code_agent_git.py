from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.base import CodeActionStatus
from jarvis.code_agent_runtime.git import GitManager
from jarvis.code_agent_runtime.tools.terminal_runner import TerminalRunner


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    if not _git_available():
        pytest.skip("git is not installed")
    _run_git(root, "init")
    _run_git(root, "config", "user.email", "jarvis@example.invalid")
    _run_git(root, "config", "user.name", "Jarvis Test")
    (root / "README.md").write_text("# Jarvis\n", encoding="utf-8")
    _run_git(root, "add", "README.md")
    _run_git(root, "commit", "-m", "init")


def test_git_status_handles_non_repo_without_crashing(tmp_path: Path) -> None:
    receipt = CodeAgentRuntimeService(tmp_path).git_status()

    assert receipt.status == CodeActionStatus.OK
    assert receipt.data["git"]["is_repo"] is False
    assert "not a git repository" in receipt.message


def test_git_status_diff_stat_and_changed_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("# Jarvis\n\nchanged\n", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    status = service.git_status()
    changed = service.git_changed_files()
    stat = service.git_diff_stat()

    assert status.status == CodeActionStatus.OK
    assert status.data["git"]["is_repo"] is True
    assert "README.md" in changed.data["git"]["changed_files"]
    assert "README.md" in stat.data["git"]["stat"]


def test_git_diff_blocks_path_outside_project(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    receipt = CodeAgentRuntimeService(tmp_path).git_diff(path=str(outside))

    assert receipt.status == CodeActionStatus.BLOCKED
    assert "outside project root" in receipt.message


def test_git_branch_name_sanitization_blocks_dangerous_names(tmp_path: Path) -> None:
    manager = GitManager(tmp_path, TerminalRunner(tmp_path))

    assert manager.sanitize_branch_name("Task Memory CLI") == "jarvis/task-memory-cli"
    assert manager.sanitize_branch_name("bad;name && thing") == "jarvis/bad-name-thing"
    assert manager.sanitize_branch_name("bad$(name)|thing>out") == "jarvis/bad-name-thing-out"
    with pytest.raises(ValueError):
        manager.sanitize_branch_name("../main")
    with pytest.raises(ValueError):
        manager.sanitize_branch_name("bad@{name")
    with pytest.raises(ValueError):
        manager.sanitize_branch_name("")
    with pytest.raises(ValueError):
        manager.sanitize_branch_name("-danger")
    with pytest.raises(ValueError):
        manager.sanitize_branch_name("C:\\temp\\branch")


def test_git_create_branch_requires_authorization_and_can_create_safe_branch(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    needs_pin = service.git_create_branch("task-fix-tests", confirm=True)
    service.configure_pin("1234")
    service.set_mode("admin")
    ok = service.git_create_branch("task-fix-tests", confirm=True, pin="1234")

    assert needs_pin.status == CodeActionStatus.CONFIRMATION_REQUIRED
    assert ok.status == CodeActionStatus.OK
    assert ok.data["git"]["branch"] == "jarvis/task-fix-tests"


def test_git_checkpoint_requires_pin_and_records_memory(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("pending", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)
    service.configure_pin("1234")
    service.set_mode("admin")

    blocked = service.git_checkpoint(message="phase4 checkpoint", confirm=True, pin="0000")
    ok = service.git_checkpoint(message="phase4 checkpoint", confirm=True, pin="1234")
    memory = service.memory_show()

    assert blocked.status == CodeActionStatus.CONFIRMATION_REQUIRED
    assert ok.status == CodeActionStatus.OK
    assert ok.data["git"]["kind"] == "stash"
    assert any(item["title"] == "Git checkpoint created" for item in memory["technical_decisions"])


def test_git_checkpoint_message_injection_is_sanitized(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("pending", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)
    service.configure_pin("1234")
    service.set_mode("admin")

    message = 'phase4 && rm -rf .; echo "secret" || whoami | sh `bad` $(bad) > out'
    receipt = service.git_checkpoint(message=message, confirm=True, pin="1234")
    memory = service.memory_show()

    assert receipt.status == CodeActionStatus.OK
    command = receipt.commands[0]
    stored_message = receipt.data["git"]["message"]
    for unsafe in ("&&", ";", "||", "|", "`", "$(", ">", '"'):
        assert unsafe not in command
        assert unsafe not in stored_message
    assert not any("secret" in str(item).casefold() for item in memory["technical_decisions"])


def test_git_dangerous_remote_and_destructive_commands_are_blocked(tmp_path: Path) -> None:
    service = CodeAgentRuntimeService(tmp_path)

    for command in ("git push", "git push --force", "git remote -v", "git reset --hard", "git clean -fd", "git rebase main", "git branch -D main", "git branch -d main"):
        receipt = service.run_command(command, dry_run=True)
        assert receipt.status == CodeActionStatus.BLOCKED, command


def test_git_revert_file_blocks_external_and_requires_auth_for_sensitive_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=value", encoding="utf-8")
    _run_git(tmp_path, "add", ".env")
    _run_git(tmp_path, "commit", "-m", "env fixture")
    service = CodeAgentRuntimeService(tmp_path)

    outside_receipt = service.git_revert_file(str(outside), confirm=True)
    sensitive_receipt = service.git_revert_file(".env", confirm=True)

    assert outside_receipt.status == CodeActionStatus.BLOCKED
    assert "outside project root" in outside_receipt.message
    assert sensitive_receipt.status == CodeActionStatus.CONFIRMATION_REQUIRED
    assert sensitive_receipt.pin_required is True


def test_git_diff_blocks_symlink_that_resolves_outside_project(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    outside = tmp_path.parent / "outside-linked.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "linked-outside.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("symlink creation is not available in this environment")

    receipt = CodeAgentRuntimeService(tmp_path).git_diff(path=str(link))

    assert receipt.status == CodeActionStatus.BLOCKED
    assert "outside project root" in receipt.message


def test_git_large_diff_is_truncated(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("\n".join(f"line-{index:05d}" for index in range(30_000)), encoding="utf-8")

    receipt = CodeAgentRuntimeService(tmp_path).git_diff()

    assert receipt.status == CodeActionStatus.OK
    assert receipt.data["git"]["truncated"] is True
    assert len(receipt.data["git"]["diff"]) <= 20_000


def test_git_memory_records_activity_without_full_diff_or_secrets(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("# Jarvis\n\nchanged\n", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    service.git_summary()
    service.configure_pin("1234")
    service.set_mode("admin")
    service.git_checkpoint(message="token secret checkpoint", confirm=True, pin="1234")
    memory = service.memory_show()
    raw = str(memory).casefold()

    assert "diff --git" not in raw
    assert "token secret checkpoint" not in raw
    assert "[redacted]" in raw


def test_git_cli_basic_status_and_changed_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("# Jarvis\n\nchanged\n", encoding="utf-8")
    runner = CliRunner()

    status = runner.invoke(app, ["code", "git", "status", "--root", str(tmp_path)])
    changed = runner.invoke(app, ["code", "git", "changed-files", "--root", str(tmp_path)])

    assert status.exit_code == 0
    assert '"is_repo": true' in status.stdout
    assert changed.exit_code == 0
    assert "README.md" in changed.stdout


def test_git_cli_non_repo_returns_useful_message(tmp_path: Path) -> None:
    runner = CliRunner()

    status = runner.invoke(app, ["code", "git", "status", "--root", str(tmp_path)])

    assert status.exit_code == 0
    assert "not a git repository" in status.stdout
    assert '"is_repo": false' in status.stdout
