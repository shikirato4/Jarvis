from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import SystemSafetyError
from jarvis.system_runtime.base import ResolvedSystemTarget, SystemOpenMode, SystemTargetKind
from jarvis.system_runtime.safeguards import (
    build_system_open_policy,
    build_system_safety_policy,
    control_advice_for_target,
    validate_allowed_target,
)


def test_system_safeguards_block_executable_extension(tmp_path: Path) -> None:
    script = tmp_path / "danger.ps1"
    script.write_text("Write-Host hi")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        system_blocked_extensions=(".ps1",),
    )
    target = ResolvedSystemTarget(target_id="1", kind=SystemTargetKind.FILE, display_name=script.name, path=str(script))
    with pytest.raises(SystemSafetyError):
        validate_allowed_target(target, policy=build_system_safety_policy(settings), open_policy=build_system_open_policy(settings))


def test_system_safeguards_require_confirmation_for_sensitive_root(tmp_path: Path) -> None:
    sensitive = tmp_path / "sensitive"
    sensitive.mkdir()
    target_file = sensitive / "secret.txt"
    target_file.write_text("secret")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        system_sensitive_roots=(sensitive,),
    )
    advice = control_advice_for_target(
        ResolvedSystemTarget(target_id="2", kind=SystemTargetKind.FILE, display_name=target_file.name, path=str(target_file)),
        policy=build_system_safety_policy(settings),
        mode=SystemOpenMode.OPEN_WITH_ASSOCIATION,
    )
    assert advice.requires_confirmation is True
    assert advice.sensitivity.value in {"high", "critical"}
