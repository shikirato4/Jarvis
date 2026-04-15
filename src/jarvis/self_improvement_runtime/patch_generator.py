from __future__ import annotations

import difflib
import logging
from pathlib import Path

from .models import SelfImprovementIssue, SelfImprovementProposal


class SelfImprovementPatchGenerator:
    def __init__(self, *, model_adapter=None, logger: logging.Logger | None = None) -> None:
        self._model_adapter = model_adapter
        self._logger = logger or logging.getLogger("jarvis.self_improvement.patch_generator")

    def propose_fix(self, analysis: dict[str, object], *, prompt: str, workspace_root: Path) -> SelfImprovementProposal | None:
        issues = [item for item in analysis.get("issues", []) if isinstance(item, SelfImprovementIssue)]
        guided = next((issue for issue in issues if issue.kind == "guided_fix" and issue.auto_fixable), None)
        if guided is not None:
            return self._proposal_from_guided_issue(guided)
        if self._model_adapter is not None:
            return self._proposal_from_model(analysis, prompt=prompt, workspace_root=workspace_root)
        return None

    def generate_patch(self, *, file_path: str, original_text: str, updated_text: str) -> str:
        diff = difflib.unified_diff(
            original_text.splitlines(),
            updated_text.splitlines(),
            fromfile=file_path,
            tofile=file_path,
            lineterm="",
        )
        return "\n".join(diff)

    def _proposal_from_guided_issue(self, issue: SelfImprovementIssue) -> SelfImprovementProposal | None:
        file_path = Path(issue.file_path)
        text = file_path.read_text(encoding="utf-8")
        original = str(issue.metadata.get("old") or "")
        updated = str(issue.metadata.get("new") or "")
        if not original or original not in text:
            return None
        replaced = False
        lines: list[str] = []
        for line in text.splitlines():
            if not replaced and "jarvis-self-improve:" not in line and original in line:
                lines.append(line.replace(original, updated, 1))
                replaced = True
                continue
            lines.append(line)
        if not replaced:
            return None
        new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        diff = self.generate_patch(file_path=str(file_path), original_text=text, updated_text=new_text)
        return SelfImprovementProposal(
            file_path=str(file_path),
            summary=issue.summary,
            rationale=issue.fix_hint or "Aplicar reemplazo guiado validado por la heurística.",
            original_text=text,
            updated_text=new_text,
            diff=diff,
            metadata={"source": "guided_fix", "issue_id": issue.issue_id},
        )

    def _proposal_from_model(self, analysis: dict[str, object], *, prompt: str, workspace_root: Path) -> SelfImprovementProposal | None:
        try:
            payload = self._model_adapter.infer_json(
                task_type="coding",
                logical_model="coding_engine",
                correlation_id="self-improvement-patch",
                metadata={"component": "self_improvement_runtime"},
                prompt=(
                    "Return JSON only with file_path, summary, rationale, original, replacement. "
                    f"User request: {prompt}\nAnalysis: {analysis}"
                ),
            )
        except Exception:  # noqa: BLE001
            payload = None
        if not payload:
            return None
        relative_path = str(payload.get("file_path") or "").strip()
        original = str(payload.get("original") or "")
        replacement = str(payload.get("replacement") or "")
        if not relative_path or not original:
            return None
        file_path = (workspace_root / relative_path).resolve()
        if workspace_root.resolve() not in file_path.parents and file_path != workspace_root.resolve():
            return None
        text = file_path.read_text(encoding="utf-8")
        if original not in text:
            return None
        updated_text = text.replace(original, replacement, 1)
        return SelfImprovementProposal(
            file_path=str(file_path),
            summary=str(payload.get("summary") or "Modelo propuso una mejora."),
            rationale=str(payload.get("rationale") or "Patch sugerido por el modelo."),
            original_text=text,
            updated_text=updated_text,
            diff=self.generate_patch(file_path=str(file_path), original_text=text, updated_text=updated_text),
            metadata={"source": "model"},
        )
