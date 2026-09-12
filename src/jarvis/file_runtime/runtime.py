from __future__ import annotations

import ctypes
import errno
import hashlib
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import re
import unicodedata
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from .models import FileErrorCode, FileOperationError, OverwritePolicy


_KNOWN_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
}


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "_GUID":
        import uuid

        raw = uuid.UUID(value).bytes_le
        return cls.from_buffer_copy(raw)


class FileNavigator:
    """Bounded, verified filesystem operations for JARVIS-owned actions."""

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        max_search_results: int = 100,
        max_search_depth: int = 8,
        search_timeout_seconds: float = 10.0,
        max_read_bytes: int = 1_000_000,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve(strict=False)
        self._event_sink = event_sink
        self.max_search_results = max(max_search_results, 1)
        self.max_search_depth = max(max_search_depth, 0)
        self.search_timeout_seconds = max(search_timeout_seconds, 0.01)
        self.max_read_bytes = max(max_read_bytes, 1)
        self._lock = RLock()

    def known_folders(self) -> dict[str, str]:
        result: dict[str, str] = {}
        fallbacks = {
            "desktop": Path.home() / "Desktop",
            "documents": Path.home() / "Documents",
            "downloads": Path.home() / "Downloads",
            "pictures": Path.home() / "Pictures",
            "videos": Path.home() / "Videos",
            "music": Path.home() / "Music",
        }
        for name, folder_id in _KNOWN_FOLDER_IDS.items():
            resolved = self._windows_known_folder(folder_id)
            result[name] = str(resolved or fallbacks[name].resolve(strict=False))
        result["workspace"] = str(self.workspace_root)
        return result

    def resolve_path(self, value: str | Path, *, base: str | Path | None = None) -> Path:
        raw = str(value).strip().strip('"')
        if not raw or "\x00" in raw:
            raise FileOperationError(FileErrorCode.INVALID_PATH, "Path is empty or invalid.")
        alias = raw.casefold().rstrip("\\/")
        known = self.known_folders()
        if alias in known:
            return Path(known[alias]).resolve(strict=False)
        expanded = Path(os.path.expandvars(raw)).expanduser()
        if not expanded.is_absolute():
            root = self.resolve_path(base) if base is not None else self.workspace_root
            expanded = root / expanded
        try:
            return expanded.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise self._map_error(exc, str(value)) from exc

    def list_directory(
        self,
        path: str | Path = "workspace",
        *,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "name",
        kind: str = "all",
        pattern: str | None = None,
    ) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.is_dir():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Directory was not found.", path=str(target))
        capped = min(max(limit, 1), 500)
        entries = []
        try:
            for child in target.iterdir():
                if kind == "files" and not child.is_file():
                    continue
                if kind == "directories" and not child.is_dir():
                    continue
                if pattern and not child.match(pattern):
                    continue
                entries.append(self._metadata(child))
        except OSError as exc:
            raise self._map_error(exc, str(target)) from exc
        key = {
            "size": lambda item: item.get("size") or 0,
            "modified": lambda item: item.get("modified_at") or "",
            "type": lambda item: item.get("type") or "",
        }.get(sort_by, lambda item: str(item.get("name", "")).casefold())
        entries.sort(key=key)
        page = entries[max(offset, 0) : max(offset, 0) + capped]
        return self._result("list", target, items=page, truncated=len(entries) > offset + capped, total=len(entries), offset=max(offset, 0), limit=capped)

    def search(
        self,
        pattern: str,
        *,
        scope: str | Path = "workspace",
        recursive: bool = True,
        max_depth: int | None = None,
        max_results: int | None = None,
        extension: str | None = None,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        root = self.resolve_path(scope)
        if not root.is_dir():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Search scope was not found.", path=str(root))
        depth_limit = min(max_depth if max_depth is not None else self.max_search_depth, self.max_search_depth)
        result_limit = min(max_results if max_results is not None else self.max_search_results, self.max_search_results)
        deadline = time.monotonic() + min(timeout_seconds or self.search_timeout_seconds, self.search_timeout_seconds)
        wanted_extension = extension.casefold() if extension else None
        if wanted_extension and not wanted_extension.startswith("."):
            wanted_extension = f".{wanted_extension}"
        self._emit("file_search_started", {"scope": str(root)})
        matches: list[dict[str, Any]] = []
        scanned = 0
        truncated = False
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            if cancel_check and cancel_check():
                raise FileOperationError(FileErrorCode.CANCELLED, "Search was cancelled.", path=str(root))
            if time.monotonic() >= deadline:
                truncated = True
                break
            current, depth = stack.pop()
            try:
                children = list(os.scandir(current))
            except OSError:
                continue
            for entry in children:
                scanned += 1
                entry_path = Path(entry.path)
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    is_link = entry.is_symlink()
                except OSError:
                    continue
                name_matches = entry_path.match(pattern) or pattern.casefold() in entry.name.casefold()
                extension_matches = wanted_extension is None or entry_path.suffix.casefold() == wanted_extension
                if name_matches and extension_matches:
                    matches.append(self._metadata(entry_path))
                    if len(matches) >= result_limit:
                        truncated = True
                        stack.clear()
                        break
                if recursive and is_dir and not is_link and depth < depth_limit:
                    stack.append((entry_path, depth + 1))
        payload = self._result("search", root, items=matches, truncated=truncated, scanned=scanned, max_depth=depth_limit, max_results=result_limit)
        self._emit("file_search_completed", {"scope": str(root), "result_count": len(matches), "truncated": truncated})
        return payload

    def search_content(
        self,
        query: str,
        *,
        scope: str | Path = "workspace",
        path: str | Path | None = None,
        max_results: int = 30,
        max_bytes: int = 300_000,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Search filenames and bounded text content with lexical-semantic ranking."""

        root = self.resolve_path(path) if path is not None else self.resolve_path(scope)
        if not root.exists():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Search target was not found.", path=str(root))
        deadline = time.monotonic() + min(timeout_seconds or self.search_timeout_seconds, self.search_timeout_seconds)
        limit = min(max(1, max_results), self.max_search_results)
        byte_limit = min(max(1, max_bytes), self.max_read_bytes)
        query_terms = self._content_terms(query)
        files = [root] if root.is_file() else self._iter_text_candidates(root)
        matches: list[dict[str, Any]] = []
        scanned = 0
        for candidate in files:
            if cancel_check and cancel_check():
                raise FileOperationError(FileErrorCode.CANCELLED, "Search was cancelled.", path=str(root))
            if time.monotonic() >= deadline:
                break
            scanned += 1
            try:
                with candidate.open("rb") as stream:
                    raw = stream.read(byte_limit + 1)
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            text = raw[:byte_limit].decode("utf-8", errors="replace")
            folded_text = self._fold_search_text(text)
            folded_name = self._fold_search_text(candidate.name)
            term_hits = [term for term in query_terms if term in folded_text]
            name_hits = [term for term in query_terms if term in folded_name]
            phrase = self._fold_search_text(query)
            exact = bool(phrase and phrase in folded_text)
            if not (term_hits or name_hits or exact):
                continue
            score = (2.0 if exact else 0.0) + len(term_hits) / max(len(query_terms), 1) + 0.5 * len(name_hits)
            first_positions = [folded_text.find(term) for term in term_hits if folded_text.find(term) >= 0]
            position = min(first_positions) if first_positions else 0
            start = max(0, position - 180)
            end = min(len(text), position + 520)
            matches.append(
                {
                    **self._metadata(candidate),
                    "score": round(score, 4),
                    "matched_terms": term_hits,
                    "excerpt": " ".join(text[start:end].split()),
                    "truncated_read": len(raw) > byte_limit,
                }
            )
        matches.sort(key=lambda item: (-float(item["score"]), str(item["path"]).casefold()))
        result = self._result("search_content", root, items=matches[:limit], truncated=len(matches) > limit or time.monotonic() >= deadline, scanned=scanned, max_results=limit)
        self._emit("file_content_search_completed", {"scope": str(root), "result_count": len(matches[:limit]), "scanned": scanned})
        return result

    def _iter_text_candidates(self, root: Path) -> list[Path]:
        allowed = {".txt", ".md", ".rst", ".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".html", ".htm", ".xml", ".log", ".ini", ".cfg"}
        result: list[Path] = []
        base_depth = len(root.parts)
        for current, directories, filenames in os.walk(root):
            depth = len(Path(current).parts) - base_depth
            if depth >= self.max_search_depth:
                directories[:] = []
            directories[:] = [name for name in directories if name not in {".git", ".venv", "node_modules", "runtime", "__pycache__"}]
            result.extend(Path(current) / name for name in filenames if Path(name).suffix.casefold() in allowed)
        return result

    @staticmethod
    def _fold_search_text(value: str) -> str:
        return "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char))

    @classmethod
    def _content_terms(cls, query: str) -> list[str]:
        ignored = {"busca", "buscar", "donde", "mis", "archivos", "archivo", "documentos", "documento", "mencione", "mencioné", "hablamos", "sobre", "todo", "relacionado"}
        return list(dict.fromkeys(term for term in re.findall(r"[a-z0-9_-]{3,}", cls._fold_search_text(query)) if term not in ignored))

    def stat(self, path: str | Path, *, sha256: bool = False) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists() and not target.is_symlink():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Path was not found.", path=str(target))
        metadata = self._metadata(target)
        if sha256 and target.is_file():
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            metadata["sha256"] = digest.hexdigest()
        return self._result("stat", target, metadata=metadata)

    def read_text(self, path: str | Path, *, encoding: str = "utf-8", max_bytes: int | None = None) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.is_file():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Text file was not found.", path=str(target))
        limit = min(max_bytes or self.max_read_bytes, self.max_read_bytes)
        with target.open("rb") as stream:
            raw = stream.read(limit + 1)
        if b"\x00" in raw[:8192]:
            raise FileOperationError(FileErrorCode.INVALID_PATH, "Binary files cannot be returned as text.", path=str(target))
        truncated = len(raw) > limit
        text = raw[:limit].decode(encoding, errors="replace")
        return self._result("read_text", target, metadata=self._metadata(target), content=text, encoding=encoding, truncated=truncated, bytes_read=min(len(raw), limit))

    def create_directory(self, path: str | Path, *, parents: bool = False, exist_ok: bool = False) -> dict[str, Any]:
        target = self.resolve_path(path)
        with self._lock:
            try:
                target.mkdir(parents=parents, exist_ok=exist_ok)
            except OSError as exc:
                raise self._map_error(exc, str(target)) from exc
            verified = target.exists() and target.is_dir()
        self._emit("directory_created", {"path": str(target), "verified": verified})
        return self._result("create_directory", target, verified=verified, metadata=self._metadata(target))

    def create_file(
        self,
        path: str | Path,
        *,
        content: str = "",
        encoding: str = "utf-8",
        overwrite: OverwritePolicy | str = OverwritePolicy.FAIL_IF_EXISTS,
        parents: bool = False,
    ) -> dict[str, Any]:
        target = self.resolve_path(path)
        policy = OverwritePolicy(overwrite)
        with self._lock:
            if parents:
                target.parent.mkdir(parents=True, exist_ok=True)
            target = self._prepare_destination(
                target,
                policy,
                preserve_file_for_atomic_replace=True,
            )
            self._atomic_write(target, content, encoding, replace=policy is OverwritePolicy.REPLACE)
            verified = target.is_file() and target.read_text(encoding=encoding) == content
        self._emit("file_created", {"path": str(target), "verified": verified})
        return self._result("create_file", target, verified=verified, metadata=self._metadata(target), artifacts=[str(target)])

    def copy(self, source: str | Path, destination: str | Path, *, overwrite: OverwritePolicy | str = OverwritePolicy.FAIL_IF_EXISTS) -> dict[str, Any]:
        src = self.resolve_path(source)
        dst = self.resolve_path(destination)
        policy = OverwritePolicy(overwrite)
        if not src.exists():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Source was not found.", path=str(src))
        if dst.is_dir() and src.is_file():
            dst = dst / src.name
        with self._lock:
            dst = self._prepare_destination(dst, policy)
            try:
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            except OSError as exc:
                raise self._map_error(exc, str(dst)) from exc
            verified = src.exists() and dst.exists() and src.is_dir() == dst.is_dir()
            if src.is_file() and dst.is_file():
                verified = verified and src.stat().st_size == dst.stat().st_size
        self._emit("file_copied", {"source": str(src), "destination": str(dst), "verified": verified})
        return self._result("copy", src, destination=dst, verified=verified, metadata=self._metadata(dst), artifacts=[str(dst)])

    def move(self, source: str | Path, destination: str | Path, *, overwrite: OverwritePolicy | str = OverwritePolicy.FAIL_IF_EXISTS) -> dict[str, Any]:
        src = self.resolve_path(source)
        dst = self.resolve_path(destination)
        policy = OverwritePolicy(overwrite)
        if not src.exists():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Source was not found.", path=str(src))
        if dst.is_dir():
            dst = dst / src.name
        with self._lock:
            dst = self._prepare_destination(dst, policy)
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
            except OSError as exc:
                raise self._map_error(exc, str(dst)) from exc
            verified = not src.exists() and dst.exists()
        self._emit("file_moved", {"source": str(src), "destination": str(dst), "verified": verified})
        return self._result("move", src, destination=dst, verified=verified, metadata=self._metadata(dst), artifacts=[str(dst)])

    def rename(self, path: str | Path, new_name: str, *, overwrite: OverwritePolicy | str = OverwritePolicy.FAIL_IF_EXISTS) -> dict[str, Any]:
        if Path(new_name).name != new_name or new_name in {"", ".", ".."}:
            raise FileOperationError(FileErrorCode.INVALID_NAME, "Rename requires a filename, not a destination path.")
        source = self.resolve_path(path)
        result = self.move(source, source.with_name(new_name), overwrite=overwrite)
        result["operation"] = "rename"
        self._emit("file_renamed", {"source": str(source), "destination": result["destination"], "verified": result["verified"]})
        return result

    def delete(self, path: str | Path, *, permanent: bool = False, recursive: bool = False) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists() and not target.is_symlink():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Path was not found.", path=str(target))
        with self._lock:
            if permanent:
                if target.is_dir() and not target.is_symlink():
                    if recursive:
                        shutil.rmtree(target)
                    else:
                        target.rmdir()
                else:
                    target.unlink()
                operation = "permanent_delete"
            else:
                self._send_to_recycle_bin(target)
                operation = "trash"
            verified = not target.exists()
        self._emit("file_trashed" if not permanent else "file_deleted", {"path": str(target), "verified": verified})
        return self._result(operation, target, verified=verified)

    def batch_move(self, items: Iterable[str | Path], destination: str | Path) -> dict[str, Any]:
        destination_path = self.resolve_path(destination)
        all_values = list(items)
        values = all_values[:100]
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for item in values:
            try:
                results.append(self.move(item, destination_path))
            except FileOperationError as exc:
                errors.append({"source": str(item), "error_code": exc.code.value, "message": str(exc)})
        return {"operation": "batch_move", "status": "partial_failure" if errors else "success", "verified": not errors, "items": results, "errors": errors, "total": len(values), "succeeded": len(results), "failed": len(errors), "truncated": len(all_values) > 100}

    def open_with_default_app(self, path: str | Path) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Path was not found.", path=str(target))
        os.startfile(str(target))  # type: ignore[attr-defined]
        return self._result("open", target, verified=True)

    def reveal(self, path: str | Path) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists():
            raise FileOperationError(FileErrorCode.NOT_FOUND, "Path was not found.", path=str(target))
        args = ["explorer.exe", f"/select,{target}"] if target.is_file() else ["explorer.exe", str(target)]
        subprocess.Popen(args, close_fds=True)
        return self._result("reveal", target, verified=True)

    def _prepare_destination(
        self,
        target: Path,
        policy: OverwritePolicy,
        *,
        preserve_file_for_atomic_replace: bool = False,
    ) -> Path:
        if not target.exists():
            return target
        if policy is OverwritePolicy.FAIL_IF_EXISTS:
            raise FileOperationError(FileErrorCode.ALREADY_EXISTS, "Destination already exists.", path=str(target))
        if policy is OverwritePolicy.RENAME_NEW:
            for index in range(1, 10_000):
                candidate = target.with_name(f"{target.stem} ({index}){target.suffix}")
                if not candidate.exists():
                    return candidate
            raise FileOperationError(FileErrorCode.ALREADY_EXISTS, "No conflict-free destination name was available.", path=str(target))
        if target.is_file() and preserve_file_for_atomic_replace:
            return target
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        return target

    @staticmethod
    def _atomic_write(target: Path, content: str, encoding: str, *, replace: bool) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists() and not replace:
                raise FileOperationError(FileErrorCode.ALREADY_EXISTS, "Destination already exists.", path=str(target))
            os.replace(temp_name, target)
        finally:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _metadata(path: Path) -> dict[str, Any]:
        stat = path.lstat()
        is_link = path.is_symlink()
        is_dir = path.is_dir() if not is_link else False
        mime, _ = mimetypes.guess_type(path.name, strict=False)
        attributes = getattr(stat, "st_file_attributes", 0)
        return {
            "name": path.name,
            "path": str(path.resolve(strict=False)),
            "exists": path.exists(),
            "type": "symlink" if is_link else "directory" if is_dir else "file",
            "is_file": path.is_file() if not is_link else False,
            "is_directory": is_dir,
            "size": stat.st_size if not is_dir else None,
            "extension": path.suffix.casefold(),
            "mime_type": mime,
            "created_at": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "readonly": not os.access(path, os.W_OK),
            "hidden": bool(attributes & 0x2) or path.name.startswith("."),
            "system": bool(attributes & 0x4),
            "reparse_point": bool(attributes & 0x400) or is_link,
        }

    @staticmethod
    def _result(operation: str, source: Path, *, destination: Path | None = None, verified: bool = True, **extra: Any) -> dict[str, Any]:
        return {"operation": operation, "source": str(source), "destination": str(destination) if destination else None, "status": "success" if verified else "failed", "verified": verified, **extra}

    def _emit(self, event_name: str, payload: dict[str, Any]) -> None:
        if self._event_sink:
            self._event_sink(event_name, payload)

    @staticmethod
    def _windows_known_folder(folder_id: str) -> Path | None:
        if os.name != "nt":
            return None
        path_pointer = ctypes.c_wchar_p()
        guid = _GUID.from_string(folder_id)
        result = ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path_pointer))
        if result != 0:
            return None
        try:
            return Path(path_pointer.value).resolve(strict=False)
        finally:
            ctypes.windll.ole32.CoTaskMemFree(path_pointer)

    @staticmethod
    def _send_to_recycle_bin(path: Path) -> None:
        if os.name != "nt":
            raise FileOperationError(FileErrorCode.IO_ERROR, "Recycle Bin is only available on Windows.", path=str(path))

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", ctypes.c_void_p), ("wFunc", ctypes.c_uint), ("pFrom", ctypes.c_wchar_p), ("pTo", ctypes.c_wchar_p), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", ctypes.c_bool), ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", ctypes.c_wchar_p)]

        operation = SHFILEOPSTRUCTW(None, 3, f"{path}\0\0", None, 0x40 | 0x10 | 0x400, False, None, None)
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
        if result or operation.fAnyOperationsAborted:
            raise FileOperationError(FileErrorCode.CANCELLED if operation.fAnyOperationsAborted else FileErrorCode.IO_ERROR, "Recycle Bin operation did not complete.", path=str(path))

    @staticmethod
    def _map_error(exc: BaseException, path: str) -> FileOperationError:
        if isinstance(exc, FileOperationError):
            return exc
        if isinstance(exc, FileNotFoundError):
            code = FileErrorCode.NOT_FOUND
        elif isinstance(exc, FileExistsError):
            code = FileErrorCode.ALREADY_EXISTS
        elif isinstance(exc, PermissionError):
            code = FileErrorCode.PERMISSION_DENIED
        elif isinstance(exc, OSError) and exc.errno == errno.ENAMETOOLONG:
            code = FileErrorCode.PATH_TOO_LONG
        elif isinstance(exc, OSError) and exc.errno == errno.ENOTEMPTY:
            code = FileErrorCode.DIRECTORY_NOT_EMPTY
        elif isinstance(exc, OSError) and exc.errno == errno.EXDEV:
            code = FileErrorCode.CROSS_DEVICE_FAILURE
        elif isinstance(exc, OSError):
            code = FileErrorCode.IO_ERROR
        else:
            code = FileErrorCode.UNKNOWN_FAILURE
        return FileOperationError(code, str(exc) or code.value, path=path)
