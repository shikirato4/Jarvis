from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.core.errors import UnitySafetyError
from jarvis.unity_runtime.base import UnityEditorOperationKind, UnityProjectDescriptor, UnityProjectStatus
from jarvis.unity_runtime.safeguards import build_unity_safety_policy, control_advice_for_operation, validate_asset_write_path, validate_project_allowed


def _project(root: Path) -> UnityProjectDescriptor:
    return UnityProjectDescriptor(
        project_id=str(root),
        name=root.name,
        project_root=str(root),
        assets_path=str(root / "Assets"),
        packages_path=str(root / "Packages"),
        project_settings_path=str(root / "ProjectSettings"),
        status=UnityProjectStatus.RESOLVED,
        is_valid_project=True,
        resolution_confidence=1.0,
    )


def test_unity_safeguards_block_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "blocked_project"
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path, unity_project_blocked_roots=(project_root,))
    with pytest.raises(UnitySafetyError):
        validate_project_allowed(_project(project_root), policy=build_unity_safety_policy(settings))


def test_unity_safeguards_confirmation_for_bridge_and_overwrite(tmp_path: Path) -> None:
    project_root = tmp_path / "game"
    advice = control_advice_for_operation(operation_kind=UnityEditorOperationKind.BRIDGE_COMMAND, project=_project(project_root), uses_bridge=True)
    assert advice.requires_confirmation is True
    overwrite = control_advice_for_operation(operation_kind=UnityEditorOperationKind.WRITE_SCRIPT, project=_project(project_root), would_overwrite=True)
    assert overwrite.requires_confirmation is True


def test_unity_safeguards_block_write_outside_assets(tmp_path: Path) -> None:
    project_root = tmp_path / "game"
    (project_root / "Assets").mkdir(parents=True)
    (project_root / "ProjectSettings").mkdir(parents=True)
    settings = Settings(data_dir=tmp_path / "runtime", workspace_root=tmp_path)
    with pytest.raises(UnitySafetyError):
        validate_asset_write_path(_project(project_root), project_root / "Outside.cs", policy=build_unity_safety_policy(settings))
