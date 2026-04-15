from __future__ import annotations

import logging
from pathlib import Path

from .models import (
    SelfImprovementComparison,
    SelfImprovementDiffStats,
    SelfImprovementMode,
    SelfImprovementPolicyReport,
    SelfImprovementProposal,
    SelfImprovementSyntaxResult,
    SelfImprovementTestResult,
)

_ALLOWED_EXTENSIONS = {".py"}
_SENSITIVE_MARKERS = (
    "bootstrap.py",
    "runtime.py",
    "config.py",
    "cli.py",
    "intent_router.py",
    "core/",
    "src/jarvis/core/",
)


class SelfImprovementValidator:
    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.self_improvement.validator")

    def validate_syntax(self, *, root: Path, changed_files: tuple[str, ...]) -> SelfImprovementSyntaxResult:
        errors: list[dict[str, object]] = []
        checked: list[str] = []
        for relative_path in changed_files:
            target = (root / relative_path).resolve()
            checked.append(str(target))
            try:
                compile(target.read_text(encoding="utf-8"), str(target), "exec")
            except SyntaxError as exc:
                errors.append({"file_path": str(target), "line": exc.lineno, "message": exc.msg})
        return SelfImprovementSyntaxResult(ok=not errors, checked_files=tuple(checked), errors=errors)

    def evaluate_proposal(
        self,
        *,
        workspace_root: Path,
        proposal: SelfImprovementProposal,
        issues: list,
        failure_history: list[str],
    ) -> SelfImprovementPolicyReport:
        file_path = Path(proposal.file_path).resolve()
        relative = str(file_path.relative_to(workspace_root.resolve())).replace("\\", "/")
        diff_stats = self._diff_stats(proposal.diff)
        reasons: list[str] = []
        warnings: list[str] = []
        sensitive_paths: list[str] = []

        if not proposal.diff.strip():
            reasons.append("El patch no generó diff; se considera ambiguo.")
        if file_path.suffix not in _ALLOWED_EXTENSIONS:
            reasons.append(f"Tipo de archivo no permitido para automejora: {file_path.suffix or '<sin extensión>'}.")
        if diff_stats.changed_files != 1:
            reasons.append("La propuesta modifica más de un archivo lógico o no pudo delimitarse con precisión.")
        if diff_stats.added_lines + diff_stats.removed_lines > 120:
            reasons.append("El cambio excede el tamaño máximo permitido para automejora segura.")
        if diff_stats.removed_lines > 40:
            reasons.append("El cambio elimina demasiado código para ser considerado seguro.")
        total_changed = max(diff_stats.added_lines + diff_stats.removed_lines, 1)
        if total_changed >= 20 and diff_stats.removed_lines / total_changed > 0.35:
            reasons.append("El patch borra demasiado código en relación con lo que añade.")
        if not issues:
            reasons.append("No hay issues concretos que justifiquen el patch.")
        if not proposal.summary.strip() or not proposal.rationale.strip():
            reasons.append("La propuesta no explica suficientemente el cambio o su causa.")

        lowered = relative.casefold()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            sensitive_paths.append(relative)
            warnings.append("El patch toca una zona sensible del sistema.")

        historical_match = any(lowered in entry.casefold().replace("\\", "/") for entry in failure_history)
        if historical_match:
            warnings.append("El patch toca una zona con historial reciente de fallos.")

        status = "pass"
        risk_level = "low"
        if reasons:
            status = "rejected"
            risk_level = "high"
        elif sensitive_paths or historical_match:
            status = "manual_review_required"
            risk_level = "high" if sensitive_paths else "medium"
        elif diff_stats.added_lines + diff_stats.removed_lines > 40:
            status = "manual_review_required"
            risk_level = "medium"
            warnings.append("El tamaño del cambio justifica revisión manual antes de aplicar.")

        return SelfImprovementPolicyReport(
            status=status,
            risk_level=risk_level,
            reasons=reasons,
            warnings=warnings,
            sensitive_paths=tuple(sensitive_paths),
            historically_problematic=historical_match,
            diff_stats=diff_stats,
        )

    def compare_results(self, *, baseline: SelfImprovementTestResult, candidate: SelfImprovementTestResult) -> SelfImprovementComparison:
        notes: list[str] = []
        baseline_green = baseline.exit_code == 0 and baseline.failed == 0 and baseline.errors == 0
        candidate_green = candidate.exit_code == 0 and candidate.failed == 0 and candidate.errors == 0
        new_failures = max(candidate.failed - baseline.failed, 0)
        new_errors = max(candidate.errors - baseline.errors, 0)
        safe = new_failures == 0 and new_errors == 0 and candidate.exit_code <= max(baseline.exit_code, 0)
        improved = candidate_green and (not baseline_green or candidate.failed < baseline.failed or candidate.errors < baseline.errors)
        if not candidate_green:
            notes.append("El candidato no dejó la suite completamente verde.")
        if new_failures:
            notes.append("Introdujo nuevas fallas de tests.")
        if new_errors:
            notes.append("Introdujo nuevos errores de ejecución.")
        if safe and candidate_green:
            notes.append("El cambio dejó la validación de tests en verde sin introducir nuevas fallas.")
        return SelfImprovementComparison(
            improved=improved,
            safe=safe,
            baseline_green=baseline_green,
            candidate_green=candidate_green,
            new_failures=new_failures,
            new_errors=new_errors,
            baseline_summary=baseline.summary,
            candidate_summary=candidate.summary,
            notes=notes,
        )

    def approve_or_reject(
        self,
        *,
        mode: SelfImprovementMode,
        policy: SelfImprovementPolicyReport,
        syntax: SelfImprovementSyntaxResult,
        import_validation,
        compileall_validation,
        comparison: SelfImprovementComparison,
        rollback_ready: bool,
    ) -> tuple[str, str]:
        if policy.status == "rejected":
            return "rejected", "El patch fue rechazado por política defensiva antes de aplicar."
        if not syntax.ok:
            return "rejected", "El cambio rompió sintaxis. Se descartó automáticamente."
        if not compileall_validation.ok:
            return "rejected", "El cambio rompió compileall en sandbox. Se descartó automáticamente."
        if not import_validation.ok:
            return "rejected", "El cambio rompió imports en sandbox. Se descartó automáticamente."
        if comparison.new_failures or comparison.new_errors or not comparison.safe:
            return "rejected", "El cambio introdujo regresiones observables en tests. Se descartó automáticamente."
        if mode == SelfImprovementMode.AUTO_APPLY_SAFE and not rollback_ready:
            return "rejected", "No existe un snapshot confiable para rollback; no se permite auto-apply."
        if not comparison.candidate_green:
            return "manual_review_required", "El candidato no dejó la suite verde; requiere revisión manual."
        if policy.status == "manual_review_required":
            return "manual_review_required", "El patch pasó validaciones técnicas, pero toca zonas o patrones de riesgo y requiere revisión manual."
        if comparison.improved or comparison.candidate_green:
            return "approved", "Recomendación: aplicar. El cambio pasó validación estricta."
        return "manual_review_required", "La evidencia no es suficiente para autoaprobar el cambio."

    @staticmethod
    def _diff_stats(diff: str) -> SelfImprovementDiffStats:
        added_lines = 0
        removed_lines = 0
        changed_files = 0
        for line in diff.splitlines():
            if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added_lines += 1
            elif line.startswith("-"):
                removed_lines += 1
        if diff.strip():
            changed_files = sum(1 for line in diff.splitlines() if line.startswith("+++ "))
        return SelfImprovementDiffStats(changed_files=changed_files, added_lines=added_lines, removed_lines=removed_lines)
