from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jarvis.code_agent_runtime.base import AuthorizationContext, CodeActionKind
from jarvis.code_agent_runtime.paths import is_sensitive_path, normalize_project_path, relative_to_root


class FileWriter:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root

    def write_text(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
        authorization: AuthorizationContext | None = None,
    ) -> dict[str, object]:
        resolved = normalize_project_path(self._root, path)
        self._require_authorization(resolved, authorization)
        if is_sensitive_path(resolved):
            raise PermissionError(f"sensitive file is blocked: {relative_to_root(self._root, resolved)}")
        existed = resolved.exists()
        if existed and not overwrite:
            raise FileExistsError(f"file already exists: {relative_to_root(self._root, resolved)}")
        backup_path: Path | None = None
        if existed:
            backup_path = self._backup_path(resolved)
        if not dry_run:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            if existed and backup_path is not None:
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_bytes(resolved.read_bytes())
            resolved.write_text(content, encoding="utf-8")
        return {
            "path": relative_to_root(self._root, resolved),
            "existed": existed,
            "backup_path": relative_to_root(self._root, backup_path) if backup_path is not None else None,
            "dry_run": dry_run,
        }

    def _require_authorization(self, resolved: Path, authorization: AuthorizationContext | None) -> None:
        if authorization is None:
            raise PermissionError("file write requires an authorization context")
        if not authorization.allowed:
            raise PermissionError("file write authorization is not allowed")
        if authorization.action != CodeActionKind.FILE_WRITE:
            raise PermissionError("authorization action does not allow file writing")
        auth_target = Path(authorization.target).expanduser().resolve(strict=False)
        if auth_target != resolved.resolve(strict=False):
            raise PermissionError("authorization target does not match file write target")
        if authorization.risk.requires_confirmation and not authorization.confirmation_confirmed:
            raise PermissionError("file write requires confirmed authorization")
        if authorization.risk.requires_pin and authorization.pin_verified is not True:
            raise PermissionError("file write requires verified PIN authorization")

    def _backup_path(self, path: Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = self._root / "runtime" / "code_agent_backups"
        rel = path.relative_to(self._root)
        safe_name = "__".join(rel.parts)
        return backup_dir / f"{safe_name}.{timestamp}.bak"
