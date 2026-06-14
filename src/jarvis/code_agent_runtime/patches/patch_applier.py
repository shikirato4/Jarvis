from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis.code_agent_runtime.base import CodeActionStatus
from jarvis.code_agent_runtime.patches.diff_utils import sha256_text
from jarvis.code_agent_runtime.patches.models import PatchApplyResult, ProposedPatch
from jarvis.code_agent_runtime.patches.patch_store import PatchStore
from jarvis.code_agent_runtime.paths import normalize_project_path

if TYPE_CHECKING:
    from jarvis.code_agent_runtime.executor import CodeAgentExecutor


class PatchApplier:
    def __init__(self, executor: "CodeAgentExecutor", store: PatchStore) -> None:
        self._executor = executor
        self._store = store

    def apply(self, patch_id: str, *, confirm: bool = False, pin: str | None = None, checkpoint: bool = False) -> dict[str, Any]:
        try:
            patch = self._store.load(patch_id)
        except ValueError as exc:
            return PatchApplyResult(patch_id=patch_id, status="failed", message=str(exc), errors=[str(exc)]).to_dict()
        if patch.status != "proposed":
            return PatchApplyResult(patch_id=patch.id, status="blocked", message=f"patch status is {patch.status}, not proposed").to_dict()
        if patch.requires_confirmation and not confirm:
            return PatchApplyResult(patch_id=patch.id, status="blocked", message="patch requires confirmation").to_dict()
        if patch.requires_pin and not pin:
            return PatchApplyResult(patch_id=patch.id, status="blocked", message="patch requires PIN").to_dict()
        conflict = self._conflict(patch)
        if conflict:
            self._store.update_status(patch.id, "failed", warning=conflict)
            return PatchApplyResult(patch_id=patch.id, status="failed", message=conflict, errors=[conflict]).to_dict()
        commands: list[str] = []
        if checkpoint:
            checkpoint_receipt = self._executor.git_checkpoint(confirm=confirm, pin=pin, message=f"jarvis patch {patch.id}")
            commands.extend(checkpoint_receipt.commands)
            if checkpoint_receipt.status in {CodeActionStatus.BLOCKED, CodeActionStatus.CONFIRMATION_REQUIRED, CodeActionStatus.FAILED}:
                self._store.update_status(patch.id, "blocked", warning=checkpoint_receipt.message)
                return PatchApplyResult(patch_id=patch.id, status="blocked", message=checkpoint_receipt.message, commands=commands).to_dict()
        touched: list[str] = []
        errors: list[str] = []
        for change in patch.changes:
            receipt = self._executor.write_file(change.path, change.new_content, overwrite=change.existed, confirm=confirm, pin=pin)
            if receipt.status != CodeActionStatus.OK:
                errors.append(receipt.message)
                self._store.update_status(patch.id, "blocked" if receipt.status in {CodeActionStatus.BLOCKED, CodeActionStatus.CONFIRMATION_REQUIRED} else "failed", warning=receipt.message)
                return PatchApplyResult(patch_id=patch.id, status="blocked" if receipt.status != CodeActionStatus.FAILED else "failed", touched_files=touched, commands=commands, message=receipt.message, errors=errors).to_dict()
            touched.extend(receipt.touched_files)
        diff_stat = self._executor.git_diff_stat().model_dump(mode="json")
        self._store.update_status(patch.id, "applied")
        return PatchApplyResult(patch_id=patch.id, status="applied", touched_files=touched, commands=commands, message="patch applied", git_diff_stat=diff_stat, errors=errors).to_dict()

    def _conflict(self, patch: ProposedPatch) -> str:
        for change in patch.changes:
            target = normalize_project_path(self._executor.project_root, change.path)
            if change.existed and not target.exists():
                return f"conflict: target file no longer exists: {change.path}"
            current = target.read_bytes().decode("utf-8", errors="replace") if target.exists() else ""
            if sha256_text(current) != change.original_hash:
                return f"conflict: target file changed since patch was proposed: {change.path}"
        return ""
