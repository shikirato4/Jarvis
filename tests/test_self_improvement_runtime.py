from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import ServiceUnavailableError
from jarvis.self_improvement_runtime.analyzer import SelfImprovementAnalyzer
from jarvis.self_improvement_runtime.models import SelfImprovementProposal, SelfImprovementRequest, SelfImprovementStatus
from jarvis.self_improvement_runtime.patch_generator import SelfImprovementPatchGenerator


def _make_workspace(
    tmp_path: Path,
    *,
    module_name: str = "calc",
    function_name: str = "add",
    original_line: str = "return a - b",
    guided_old: str = "return a - b",
    guided_new: str = "return a + b",
    expected: int = 5,
) -> Path:
    workspace = tmp_path / "project"
    workspace.mkdir()
    module_path = workspace / f"{module_name}.py"
    module_path.write_text(
        f"def {function_name}(a, b):\n"
        f'    # jarvis-self-improve: replace "{guided_old}" => "{guided_new}"\n'
        f"    {original_line}\n",
        encoding="utf-8",
    )
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / f"test_{module_name}.py").write_text(
        f"from {module_name} import {function_name}\n\n"
        f"def test_{function_name}():\n"
        f"    assert {function_name}(2, 3) == {expected}\n",
        encoding="utf-8",
    )
    return workspace


def _build_app(tmp_path: Path, workspace: Path | None = None):
    workspace = workspace or _make_workspace(tmp_path)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=workspace,
        research_allowed_roots=(workspace,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )
    app = build_application(settings)
    app.start()
    return app, workspace


def _proposal_for(file_path: Path, updated_text: str, *, summary: str = "Cambio propuesto", rationale: str = "Corrección puntual") -> SelfImprovementProposal:
    original_text = file_path.read_text(encoding="utf-8")
    generator = SelfImprovementPatchGenerator()
    return SelfImprovementProposal(
        file_path=str(file_path),
        summary=summary,
        rationale=rationale,
        original_text=original_text,
        updated_text=updated_text,
        diff=generator.generate_patch(file_path=str(file_path), original_text=original_text, updated_text=updated_text),
        metadata={"source": "test"},
    )


def _override_proposal(app, proposal: SelfImprovementProposal) -> None:
    app.self_improvement_runtime_service._patch_generator.propose_fix = lambda analysis, prompt, workspace_root: proposal  # noqa: SLF001


def test_analyzer_detects_guided_issue(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    analyzer = SelfImprovementAnalyzer()
    analysis = analyzer.analyze_code(workspace / "calc.py")

    assert analysis["issues"]
    assert analysis["issues"][0].kind == "guided_fix"
    assert analysis["issues"][0].auto_fixable is True


def test_patch_generator_builds_diff_from_guided_issue(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    analyzer = SelfImprovementAnalyzer()
    analysis = analyzer.analyze_code(workspace / "calc.py")
    generator = SelfImprovementPatchGenerator()

    proposal = generator.propose_fix(analysis, prompt="encuentra bugs", workspace_root=workspace)

    assert proposal is not None
    assert "return a + b" in proposal.updated_text
    assert "---" in proposal.diff and "+++" in proposal.diff


def test_self_improvement_runs_in_sandbox_and_keeps_workspace_unchanged_until_apply(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(
                prompt="encuentra bugs",
                path="calc.py",
                test_targets=("tests/test_calc.py",),
            )
        )
        assert receipt.status == SelfImprovementStatus.VALIDATED
        assert receipt.approval_decision == "approved"
        assert receipt.baseline_tests is not None and receipt.baseline_tests.failed == 1
        assert receipt.candidate_tests is not None and receipt.candidate_tests.exit_code == 0
        assert receipt.policy is not None and receipt.policy.risk_level == "low"
        assert receipt.import_validation is not None and receipt.import_validation.ok is True
        assert receipt.compileall_validation is not None and receipt.compileall_validation.ok is True
        assert "return a - b" in (workspace / "calc.py").read_text(encoding="utf-8")
        assert receipt.sandbox is not None
        sandbox_calc = Path(receipt.sandbox.sandbox_root) / "calc.py"
        assert "return a + b" in sandbox_calc.read_text(encoding="utf-8")
    finally:
        app.stop()


def test_self_improvement_auto_apply_creates_backup_and_supports_rollback(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(
                prompt="mejora el sistema",
                path="calc.py",
                test_targets=("tests/test_calc.py",),
                auto_apply=True,
                require_confirmation=False,
            )
        )
        assert receipt.status == SelfImprovementStatus.APPLIED
        assert receipt.backup_manifest is not None
        history_path = tmp_path / "runtime" / "self_improvement" / "history" / f"{receipt.session_id}.json"
        manifest = json.loads(history_path.read_text(encoding="utf-8"))
        assert manifest["applied"] is True
        assert "return a + b" in (workspace / "calc.py").read_text(encoding="utf-8")

        rollback = app.runtime_service.self_improvement_rollback(receipt.session_id)
        assert rollback.status == SelfImprovementStatus.ROLLED_BACK
        assert "return a - b" in (workspace / "calc.py").read_text(encoding="utf-8")
    finally:
        app.stop()


def test_self_improvement_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    app, _workspace = _build_app(tmp_path)
    try:
        with pytest.raises(ServiceUnavailableError):
            app.runtime_service.self_improvement_analyze(SelfImprovementRequest(prompt="revisa el código", path="../fuera.py"))
    finally:
        app.stop()


def test_self_improvement_rejects_overly_large_patch(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        calc_path = workspace / "calc.py"
        noisy_block = "\n".join(f"    value_{index} = a + b" for index in range(140))
        updated_text = (
            "def add(a, b):\n"
            f"{noisy_block}\n"
            "    return a + b\n"
        )
        _override_proposal(app, _proposal_for(calc_path, updated_text))

        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="optimiza este módulo", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert receipt.status == SelfImprovementStatus.REJECTED
        assert receipt.approval_decision == "rejected"
        assert receipt.policy is not None
        assert any("tamaño máximo" in reason for reason in receipt.policy.reasons)
    finally:
        app.stop()


def test_self_improvement_requires_manual_review_for_sensitive_files(tmp_path: Path) -> None:
    workspace = _make_workspace(
        tmp_path,
        module_name="bootstrap",
        function_name="boot",
        original_line="return a - b",
        guided_old="return a - b",
        guided_new="return a + b",
        expected=5,
    )
    app, _workspace = _build_app(tmp_path, workspace)
    try:
        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(
                prompt="mejora el sistema",
                path="bootstrap.py",
                test_targets=("tests/test_bootstrap.py",),
                auto_apply=True,
                require_confirmation=False,
            )
        )

        assert receipt.status == SelfImprovementStatus.VALIDATED
        assert receipt.approval_decision == "manual_review_required"
        assert receipt.applied is False
        assert receipt.policy is not None
        assert receipt.policy.sensitive_paths == ("bootstrap.py",)
    finally:
        app.stop()


def test_self_improvement_rejects_patch_that_introduces_new_failures(tmp_path: Path) -> None:
    workspace = _make_workspace(
        tmp_path,
        original_line="return a + b",
        guided_old="return a + b",
        guided_new="return a * b",
        expected=5,
    )
    app, _workspace = _build_app(tmp_path, workspace)
    try:
        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="encuentra bugs", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert receipt.status == SelfImprovementStatus.REJECTED
        assert receipt.approval_decision == "rejected"
        assert receipt.comparison is not None and receipt.comparison.new_failures == 1
    finally:
        app.stop()


def test_self_improvement_rejects_ambiguous_patch_without_rationale(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        calc_path = workspace / "calc.py"
        proposal = _proposal_for(calc_path, calc_path.read_text(encoding="utf-8").replace("return a - b", "return a + b"))
        proposal.summary = ""
        proposal.rationale = ""
        _override_proposal(app, proposal)

        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="revisa el código", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert receipt.status == SelfImprovementStatus.REJECTED
        assert receipt.policy is not None
        assert any("no explica suficientemente" in reason for reason in receipt.policy.reasons)
    finally:
        app.stop()


def test_self_improvement_rejects_patch_that_breaks_compileall(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        calc_path = workspace / "calc.py"
        broken_text = "def add(a, b):\n    return (\n"
        _override_proposal(app, _proposal_for(calc_path, broken_text))

        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="intenta corregir esto", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert receipt.status == SelfImprovementStatus.REJECTED
        assert receipt.compileall_validation is not None and receipt.compileall_validation.ok is False
        assert "compileall" in receipt.message.casefold() or "sintaxis" in receipt.message.casefold()
    finally:
        app.stop()


def test_self_improvement_rejects_patch_that_breaks_imports(tmp_path: Path) -> None:
    app, workspace = _build_app(tmp_path)
    try:
        calc_path = workspace / "calc.py"
        broken_import_text = 'raise RuntimeError("boom")\n\n\ndef add(a, b):\n    return a + b\n'
        _override_proposal(app, _proposal_for(calc_path, broken_import_text))

        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="intenta corregir esto", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert receipt.status == SelfImprovementStatus.REJECTED
        assert receipt.import_validation is not None and receipt.import_validation.ok is False
        assert receipt.applied is False
    finally:
        app.stop()


def test_self_improvement_receipt_explains_reason_for_decision(tmp_path: Path) -> None:
    app, _workspace = _build_app(tmp_path)
    try:
        receipt = app.runtime_service.self_improvement_run(
            SelfImprovementRequest(prompt="encuentra bugs", path="calc.py", test_targets=("tests/test_calc.py",))
        )

        assert "validación" in receipt.message.casefold() or "apliqué" in receipt.message.casefold()
        assert receipt.data["validation_summary"]["syntax_ok"] is True
        assert receipt.data["validation_summary"]["candidate_green"] is True
    finally:
        app.stop()


def test_runtime_snapshot_includes_self_improvement_service(jarvis_app) -> None:
    snapshot = jarvis_app.runtime_service.snapshot()
    service_names = {service.name for service in snapshot.services}
    assert "self_improvement_runtime" in service_names
