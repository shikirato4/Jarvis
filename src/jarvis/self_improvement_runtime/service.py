from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.models import HealthStatus, ServiceStatus
from jarvis.core.services import RuntimeServiceContract

from .analyzer import SelfImprovementAnalyzer
from .executor import SelfImprovementExecutor
from .models import SelfImprovementMode, SelfImprovementReceipt, SelfImprovementRequest, SelfImprovementStatus
from .patch_generator import SelfImprovementPatchGenerator
from .sandbox import SelfImprovementSandbox
from .validator import SelfImprovementValidator


class SelfImprovementRuntimeService(RuntimeServiceContract):
    service_name = "self_improvement_runtime"

    def __init__(
        self,
        *,
        settings,
        analyzer: SelfImprovementAnalyzer,
        patch_generator: SelfImprovementPatchGenerator,
        sandbox: SelfImprovementSandbox,
        validator: SelfImprovementValidator,
        executor: SelfImprovementExecutor,
        logger: logging.Logger | None = None,
        operation_registry=None,
    ) -> None:
        self._settings = settings
        self._analyzer = analyzer
        self._patch_generator = patch_generator
        self._sandbox = sandbox
        self._validator = validator
        self._executor = executor
        self._logger = logger or logging.getLogger("jarvis.self_improvement")
        self._operations = operation_registry
        self._started = False
        self._last_receipt: SelfImprovementReceipt | None = None

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def health(self) -> ServiceStatus:
        return ServiceStatus(name=self.service_name, status=HealthStatus.READY if self._started else HealthStatus.STOPPED, details=self.status())

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "started": self._started,
            "last_session_id": self._last_receipt.session_id if self._last_receipt else None,
            "last_status": self._last_receipt.status.value if self._last_receipt else None,
            "last_decision": self._last_receipt.approval_decision if self._last_receipt else None,
            "last_mode": self._last_receipt.mode.value if self._last_receipt else None,
        }

    def analyze_code(self, request: SelfImprovementRequest | dict) -> SelfImprovementReceipt:
        self._ensure_started()
        payload = SelfImprovementRequest.model_validate(request)
        target = self._resolve_target(payload.path)
        analysis = self._analyzer.analyze_code(target)
        issues = list(analysis["issues"])[: payload.max_issues]
        message = "No detecté problemas autoaplicables, pero dejé el análisis listo."
        if issues:
            message = f"Detecté un problema potencial en {Path(issues[0].file_path).name}; requiere validación disciplinada."
        receipt = SelfImprovementReceipt(
            session_id=f"self-improve-{uuid4().hex[:8]}",
            status=SelfImprovementStatus.ANALYZED,
            prompt=payload.prompt,
            analyzed_path=str(target),
            issues=issues,
            mode=SelfImprovementMode.ANALYZE_ONLY,
            message=message,
            data={
                "failure_history": analysis.get("failure_history", []),
                "research_context": analysis.get("research_context", {}),
            },
        )
        self._persist_and_store(receipt)
        return receipt

    def run(self, request: SelfImprovementRequest | dict) -> SelfImprovementReceipt:
        self._ensure_started()
        payload = SelfImprovementRequest.model_validate(request)
        target = self._resolve_target(payload.path)
        session_id = f"self-improve-{uuid4().hex[:8]}"
        mode = self._resolve_mode(payload)
        analysis = self._analyzer.analyze_code(target)
        issues = list(analysis["issues"])[: payload.max_issues]
        failure_history = list(analysis.get("failure_history", []))
        proposal = self._patch_generator.propose_fix(analysis, prompt=payload.prompt, workspace_root=self._settings.resolved_workspace_root)
        if proposal is None:
            receipt = SelfImprovementReceipt(
                session_id=session_id,
                status=SelfImprovementStatus.REJECTED,
                prompt=payload.prompt,
                analyzed_path=str(target),
                issues=issues,
                mode=mode,
                message="No encontré un patch suficientemente acotado y seguro. Recomendación: revisión manual.",
                approval_decision="rejected",
                data={
                    "failure_history": failure_history,
                    "research_context": analysis.get("research_context", {}),
                },
            )
            self._persist_and_store(receipt)
            return receipt

        policy = self._validator.evaluate_proposal(
            workspace_root=self._settings.resolved_workspace_root,
            proposal=proposal,
            issues=issues,
            failure_history=failure_history,
        )

        sandbox_root = self._sandbox.create(workspace_root=self._settings.resolved_workspace_root, session_id=session_id)
        sandbox_record = self._sandbox.apply_patch_sandbox(
            sandbox_root=sandbox_root,
            workspace_root=self._settings.resolved_workspace_root,
            proposal=proposal,
        )
        syntax = self._validator.validate_syntax(root=sandbox_root, changed_files=sandbox_record.changed_files)
        compileall_validation = self._executor.run_compileall(cwd=sandbox_root, changed_files=sandbox_record.changed_files)
        import_validation = self._executor.run_import_checks(cwd=sandbox_root, changed_files=sandbox_record.changed_files)
        test_targets = payload.test_targets or ("tests",)
        baseline = self._executor.run_tests(cwd=self._settings.resolved_workspace_root, test_targets=test_targets)
        candidate = self._executor.run_tests(cwd=sandbox_root, test_targets=test_targets)
        comparison = self._validator.compare_results(baseline=baseline, candidate=candidate)

        backup_manifest = None
        rollback_ready = False
        if mode == SelfImprovementMode.AUTO_APPLY_SAFE:
            backup_manifest = self._sandbox.persist_backup(
                session_id=session_id,
                workspace_root=self._settings.resolved_workspace_root,
                proposal=proposal,
            )
            rollback_ready = self._sandbox.backup_ready(session_id)

        decision, message = self._validator.approve_or_reject(
            mode=mode,
            policy=policy,
            syntax=syntax,
            import_validation=import_validation,
            compileall_validation=compileall_validation,
            comparison=comparison,
            rollback_ready=rollback_ready,
        )

        status = SelfImprovementStatus.VALIDATED if decision in {"approved", "manual_review_required"} else SelfImprovementStatus.REJECTED
        applied = False
        rollback_available = rollback_ready

        if decision == "approved" and mode == SelfImprovementMode.AUTO_APPLY_SAFE:
            try:
                self._sandbox.apply_to_workspace(
                    session_id=session_id,
                    workspace_root=self._settings.resolved_workspace_root,
                    proposal=proposal,
                )
                status = SelfImprovementStatus.APPLIED
                applied = True
                rollback_available = True
                message = "Apliqué el cambio porque pasó política estricta, validación multicapa y quedó rollback listo."
            except Exception as exc:  # noqa: BLE001
                status = SelfImprovementStatus.FAILED
                applied = False
                rollback_available = False
                decision = "rejected"
                message = f"La aplicación falló y se revirtió al backup. Error: {exc}"

        receipt = SelfImprovementReceipt(
            session_id=session_id,
            status=status,
            prompt=payload.prompt,
            analyzed_path=str(target),
            issues=issues,
            proposal=proposal,
            mode=mode,
            policy=policy,
            sandbox=sandbox_record,
            syntax=syntax,
            import_validation=import_validation,
            compileall_validation=compileall_validation,
            baseline_tests=baseline,
            candidate_tests=candidate,
            comparison=comparison,
            approval_decision=decision,
            applied=applied,
            rollback_available=rollback_available,
            backup_manifest=backup_manifest,
            message=message,
            data={
                "test_targets": list(test_targets),
                "failure_history": failure_history,
                "research_context": analysis.get("research_context", {}),
                "validation_summary": {
                    "syntax_ok": syntax.ok,
                    "compileall_ok": compileall_validation.ok,
                    "imports_ok": import_validation.ok,
                    "candidate_green": comparison.candidate_green,
                },
            },
        )
        self._persist_and_store(receipt)
        return receipt

    def rollback(self, session_id: str) -> SelfImprovementReceipt:
        self._ensure_started()
        result = self._sandbox.rollback(session_id)
        receipt = SelfImprovementReceipt(
            session_id=session_id,
            status=SelfImprovementStatus.ROLLED_BACK,
            prompt="rollback",
            analyzed_path=str(self._settings.resolved_workspace_root),
            mode=SelfImprovementMode.AUTO_APPLY_SAFE,
            applied=False,
            rollback_available=False,
            message=f"Rollback completado. Restaurados: {result['restored']}.",
            data=result,
        )
        self._persist_and_store(receipt)
        return receipt

    def _resolve_target(self, raw_path: str | None) -> Path:
        if not raw_path:
            default_target = self._settings.resolved_workspace_root / "src" / "jarvis"
            return default_target if default_target.exists() else self._settings.resolved_workspace_root
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self._settings.resolved_workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        workspace_root = self._settings.resolved_workspace_root
        if workspace_root not in candidate.parents and candidate != workspace_root:
            raise ServiceUnavailableError("self improvement path must stay inside workspace", details={"path": str(candidate)})
        return candidate

    def _persist_receipt(self, receipt: SelfImprovementReceipt) -> None:
        history_dir = self._settings.resolved_data_dir / "self_improvement" / "receipts"
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / f"{receipt.session_id}.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")

    def _persist_and_store(self, receipt: SelfImprovementReceipt) -> None:
        self._persist_receipt(receipt)
        self._last_receipt = receipt

    @staticmethod
    def _resolve_mode(payload: SelfImprovementRequest) -> SelfImprovementMode:
        if payload.mode is not None:
            return payload.mode
        if payload.auto_apply and not payload.require_confirmation:
            return SelfImprovementMode.AUTO_APPLY_SAFE
        return SelfImprovementMode.SANDBOX_VALIDATED

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("self improvement runtime is not started")
