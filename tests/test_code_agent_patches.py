from __future__ import annotations

import inspect
import os
from pathlib import Path

from typer.testing import CliRunner

from jarvis.cli import app
from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.code_agent_runtime.patches import PatchApplier


def _make_project(root: Path) -> Path:
    (root / "src").mkdir(parents=True)
    target = root / "src" / "app.py"
    target.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    return target


def test_patch_propose_replace_does_not_modify_file_and_has_diff(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    before = target.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    patch = service.patch_propose_replace("src/app.py", "hello", "hola")

    assert patch["status"] == "proposed"
    assert target.read_text(encoding="utf-8") == before
    assert "--- a/src\\app.py" in patch["unified_diff"] or "--- a/src/app.py" in patch["unified_diff"]
    assert "+    return 'hola'" in patch["unified_diff"]
    assert patch["requires_confirmation"] is True


def test_patch_propose_insert_and_create_do_not_apply(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    insert = service.patch_propose_insert_after("src/app.py", "def greet():", "\n    # greeting")
    create = service.patch_propose_create_file("src/new.py", "VALUE = 1\n")

    assert insert["status"] == "proposed"
    assert "# greeting" not in target.read_text(encoding="utf-8")
    assert create["status"] == "proposed"
    assert not (tmp_path / "src" / "new.py").exists()


def test_patch_propose_insert_before_append_and_missing_anchors(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    before = target.read_text(encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    insert_before = service.patch_propose_insert_before("src/app.py", "def greet():", "# module\n")
    append = service.patch_propose_append("src/app.py", "\n# end\n")
    missing_old = service.patch_propose_replace("src/app.py", "does-not-exist", "x")
    missing_anchor = service.patch_propose_insert_after("src/app.py", "does-not-exist", "x")

    assert insert_before["status"] == "proposed"
    assert append["status"] == "proposed"
    assert target.read_text(encoding="utf-8") == before
    assert missing_old["status"] == "blocked"
    assert "old_text" in missing_old["message"]
    assert missing_anchor["status"] == "blocked"
    assert "anchor" in missing_anchor["message"]


def test_patch_create_existing_and_unified_diff_are_blocked(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    existing = service.patch_propose_create_file("src/app.py", "x = 1\n")
    arbitrary_diff = service.patch_propose_unified_diff("--- a/src/app.py\n+++ b/src/app.py\n@@\n-x\n+y\n")

    assert existing["status"] == "blocked"
    assert "already exists" in existing["message"]
    assert arbitrary_diff["status"] == "blocked"
    assert "not enabled" in arbitrary_diff["message"] or "parser" in arbitrary_diff["message"]


def test_patch_blocks_outside_sensitive_and_secret_content(tmp_path: Path) -> None:
    _make_project(tmp_path)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("x = 1", encoding="utf-8")
    service = CodeAgentRuntimeService(tmp_path)

    outside_result = service.patch_propose_replace(str(outside), "x", "y")
    sensitive = service.patch_propose_replace(".env", "TOKEN", "X")
    secret = service.patch_propose_create_file("src/secret.py", "API_KEY = 'secret-token'\n")

    assert outside_result["status"] == "blocked"
    assert sensitive["status"] == "blocked"
    assert secret["status"] == "blocked"


def test_patch_blocks_symlink_outside_project(tmp_path: Path) -> None:
    _make_project(tmp_path)
    outside = tmp_path.parent / "outside-target.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "src" / "linked.py"
    try:
        os.symlink(outside, link)
    except OSError:
        return
    service = CodeAgentRuntimeService(tmp_path)

    result = service.patch_propose_replace("src/linked.py", "x", "y")

    assert result["status"] == "blocked"


def test_patch_large_change_warns(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    large = "x = 1\n" * 5000

    patch = service.patch_propose_create_file("src/large.py", large)

    assert patch["status"] == "proposed"
    assert patch["warnings"]
    assert patch["requires_confirmation"] is True
    assert "[diff truncated]" in patch["unified_diff"] or "large patch" in str(patch["warnings"]).casefold()


def test_patch_apply_requires_confirmation_for_existing_file_and_modifies_when_confirmed(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    patch = service.patch_propose_replace("src/app.py", "hello", "hola")

    blocked = service.patch_apply(patch["id"])
    applied = service.patch_apply(patch["id"], confirm=True)

    assert blocked["status"] == "blocked"
    assert applied["status"] == "applied"
    assert "hola" in target.read_text(encoding="utf-8")
    assert "src/app.py" in {item.replace("\\", "/") for item in applied["touched_files"]}
    assert "git_diff_stat" in applied


def test_patch_conflict_when_file_changed_after_proposal(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    patch = service.patch_propose_replace("src/app.py", "hello", "hola")
    target.write_text("def greet():\n    return 'changed'\n", encoding="utf-8")

    result = service.patch_apply(patch["id"], confirm=True)

    assert result["status"] == "failed"
    assert "conflict" in result["message"]


def test_patch_store_handles_corrupt_patch_file(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    patches_dir = tmp_path / "runtime" / "code_agent" / "patches"
    patches_dir.mkdir(parents=True)
    (patches_dir / "broken.json").write_text("{broken", encoding="utf-8")

    listing = service.patch_list()

    assert listing["warnings"]
    assert list(patches_dir.glob("broken.json.corrupt-*.bak"))


def test_patch_json_does_not_store_secrets(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    blocked = service.patch_propose_create_file("src/secret_holder.py", "PASSWORD='secret-token'\n")
    patch = service.patch_propose_create_file("src/safe.py", "VALUE = 1\n")
    raw = (tmp_path / "runtime" / "code_agent" / "patches" / f"{patch['id']}.json").read_text(encoding="utf-8").casefold()

    assert blocked["status"] == "blocked"
    assert "secret-token" not in raw
    assert "password" not in raw


def test_patch_memory_without_secrets_and_no_direct_filewriter(tmp_path: Path) -> None:
    _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)

    patch = service.patch_propose_replace("src/app.py", "hello", "hola", task="replace token password .env")
    service.patch_apply(patch["id"], confirm=True)
    memory = service.memory_show()
    raw = str(memory).casefold()
    source = inspect.getsource(PatchApplier)

    assert memory["patch_events"]
    assert "[redacted]" in raw
    assert ".env" not in raw
    assert "token password" not in raw
    assert "FileWriter(" not in source
    assert ".writer.write_text(" not in source
    assert ".write_file(" in source


def test_agent_patch_integration_dry_run_assisted_apply(tmp_path: Path) -> None:
    target = _make_project(tmp_path)
    service = CodeAgentRuntimeService(tmp_path)
    patch = service.patch_propose_replace("src/app.py", "hello", "hola")

    dry_run = service.agent_run("aplica patch explícito", mode="dry-run", patch_id=patch["id"])
    assisted = service.agent_run("aplica patch explícito", mode="assisted", patch_id=patch["id"])

    assert dry_run["patch"]["id"] == patch["id"]
    assert "hola" not in target.read_text(encoding="utf-8")
    assert assisted["patch"]["status"] == "blocked"
    assert "hola" not in target.read_text(encoding="utf-8")
    applied = service.agent_run("aplica patch explícito", mode="apply", patch_id=patch["id"], confirm=True)
    assert applied["patch"]["status"] == "applied"
    assert "hola" in target.read_text(encoding="utf-8")


def test_patch_cli_propose_list_show_apply_reject_stats(tmp_path: Path) -> None:
    _make_project(tmp_path)
    runner = CliRunner()

    propose = runner.invoke(app, ["code", "patch", "propose-create-file", "--root", str(tmp_path), "--file", "src/cli_patch.py", "--content", "VALUE = 1\n"])
    assert propose.exit_code == 0
    patch_id = __import__("json").loads(propose.stdout)["id"]

    listed = runner.invoke(app, ["code", "patch", "list", "--root", str(tmp_path)])
    shown = runner.invoke(app, ["code", "patch", "show", patch_id, "--root", str(tmp_path)])
    applied = runner.invoke(app, ["code", "patch", "apply", patch_id, "--root", str(tmp_path), "--confirm"])
    stats = runner.invoke(app, ["code", "patch", "stats", "--root", str(tmp_path)])
    unified = runner.invoke(app, ["code", "patch", "propose-unified-diff", "--root", str(tmp_path), "--diff", "--- a/x\n+++ b/x\n"])

    propose_reject = runner.invoke(app, ["code", "patch", "propose-create-file", "--root", str(tmp_path), "--file", "src/reject_me.py", "--content", "VALUE = 2\n"])
    reject_id = __import__("json").loads(propose_reject.stdout)["id"]
    rejected = runner.invoke(app, ["code", "patch", "reject", reject_id, "--root", str(tmp_path)])

    assert listed.exit_code == 0
    assert patch_id in listed.stdout
    assert shown.exit_code == 0
    assert '"unified_diff"' in shown.stdout
    assert applied.exit_code == 0
    assert '"applied"' in applied.stdout
    assert (tmp_path / "src" / "cli_patch.py").exists()
    assert stats.exit_code == 0
    assert unified.exit_code == 0
    assert '"blocked"' in unified.stdout
    assert rejected.exit_code == 0
    assert '"rejected"' in rejected.stdout
