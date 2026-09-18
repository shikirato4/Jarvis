from __future__ import annotations

import ctypes
import importlib
import os
import platform
import shutil
import subprocess
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Protocol

from .models import (
    ProcessRef,
    WindowBounds,
    WindowRef,
    WindowTarget,
    WindowsEnvironment,
    WindowsErrorCode,
    WindowsObservation,
    WindowsRuntimeError,
)


class WindowsControlBackend(Protocol):
    """Provider-neutral semantic boundary for Windows inspection and control."""

    def inspect_environment(self) -> WindowsObservation: ...

    def list_processes(self, *, limit: int, interactive_only: bool) -> WindowsObservation: ...

    def list_windows(self, *, limit: int) -> WindowsObservation: ...

    def foreground_window(self) -> WindowsObservation: ...

    def resolve_window(self, target: WindowTarget) -> WindowRef: ...

    def focus_window(self, target: WindowTarget, *, timeout_seconds: float) -> WindowsObservation: ...

    def minimize_window(self, target: WindowTarget, *, timeout_seconds: float) -> WindowsObservation: ...

    def maximize_window(self, target: WindowTarget, *, timeout_seconds: float) -> WindowsObservation: ...

    def restore_window(self, target: WindowTarget, *, timeout_seconds: float) -> WindowsObservation: ...

    def launch_application(self, application: str, *, timeout_seconds: float) -> WindowsObservation: ...


_APP_ALIASES = {
    "bloc de notas": "notepad.exe",
    "notepad": "notepad.exe",
    "notepad.exe": "notepad.exe",
    "calculadora": "calculatorapp.exe",
    "calculator": "calculatorapp.exe",
    "calc": "calculatorapp.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "visual studio code": "code.exe",
    "vscode": "code.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "explorador de archivos": "explorer.exe",
    "file explorer": "explorer.exe",
}

_LAUNCH_COMMANDS = {
    "notepad.exe": ("notepad.exe",),
    "calculatorapp.exe": ("calc.exe",),
    "mspaint.exe": ("mspaint.exe",),
}

_HELPER_CLASSES = {"Progman", "WorkerW", "Cua.AgentCursorOverlay"}
_HELPER_PROCESSES = {"dwm.exe", "werfault.exe", "textinputhost.exe"}


@dataclass(frozen=True, slots=True)
class WindowCandidateState:
    exists: bool
    title: str
    process_name: str | None
    class_name: str | None
    visible: bool
    minimized: bool
    maximized: bool
    owner_handle: int
    tool_window: bool
    cloaked: bool


@dataclass(frozen=True, slots=True)
class WindowClassification:
    targetable: bool
    list_eligible: bool
    suppression_reasons: tuple[str, ...]


def classify_window_candidate(state: WindowCandidateState) -> WindowClassification:
    """Classify OS state without collapsing minimized into nonexistent."""
    reasons: list[str] = []
    if not state.exists:
        reasons.append("invalid_handle")
    if not state.title.strip():
        reasons.append("empty_title")
    if not state.visible and not state.minimized:
        reasons.append("hidden_non_minimized")
    if state.class_name in _HELPER_CLASSES:
        reasons.append("helper_class")
    if (state.process_name or "").casefold() in _HELPER_PROCESSES:
        reasons.append("helper_process")
    if state.tool_window:
        reasons.append("tool_window")
    if state.cloaked and not state.minimized:
        reasons.append("cloaked")
    if state.owner_handle:
        reasons.append("owned_window")
    targetable = not reasons
    return WindowClassification(
        targetable=targetable,
        list_eligible=targetable,
        suppression_reasons=tuple(reasons),
    )


class NativeWindowsControlBackend:
    """Win32/ctypes backend. No shell strings, elevation, or arbitrary commands."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise WindowsRuntimeError(
                WindowsErrorCode.OPERATION_NOT_SUPPORTED,
                "native Windows control is available only on Windows",
            )
        self._user32 = ctypes.windll.user32

    def inspect_environment(self) -> WindowsObservation:
        edition, version, build = self._windows_version()
        session_id = self._session_id(os.getpid())
        environment = WindowsEnvironment(
            edition=edition,
            version=version,
            build=build,
            architecture=platform.machine() or "unknown",
            interactive_session=bool(self._user32.GetDesktopWindow()),
            session_id=session_id,
        )
        return WindowsObservation(
            operation="environment.inspect",
            verified=True,
            environment=environment,
        )

    def list_processes(self, *, limit: int = 30, interactive_only: bool = False) -> WindowsObservation:
        bounded_limit = max(1, min(limit, 100))
        windows = self._enumerate_windows(purpose="target")
        visible_counts: dict[int, int] = {}
        for window in windows:
            visible_counts[window.pid] = visible_counts.get(window.pid, 0) + 1
        processes = self._processes(visible_counts, interactive_only=interactive_only)
        processes.sort(key=lambda item: (-int(item.interactive), item.process_name.casefold(), item.pid))
        return WindowsObservation(
            operation="process.list",
            verified=True,
            processes=processes[:bounded_limit],
            count=len(processes),
            truncated=len(processes) > bounded_limit,
        )

    def list_windows(self, *, limit: int = 30) -> WindowsObservation:
        bounded_limit = max(1, min(limit, 100))
        windows = self._enumerate_windows(purpose="list")
        windows.sort(key=lambda item: (not item.foreground, item.minimized, item.title.casefold()))
        return WindowsObservation(
            operation="window.list",
            verified=True,
            windows=windows[:bounded_limit],
            count=len(windows),
            truncated=len(windows) > bounded_limit,
        )

    def foreground_window(self) -> WindowsObservation:
        hwnd = int(self._user32.GetForegroundWindow() or 0)
        window = self._window_ref(hwnd) if hwnd else None
        return WindowsObservation(
            operation="window.foreground",
            verified=window is not None,
            target=window,
            message=None if window else "no foreground application window is available",
        )

    def resolve_window(self, target: WindowTarget) -> WindowRef:
        if target.runtime_id:
            try:
                return self._resolve_runtime_id(target.runtime_id)
            except WindowsRuntimeError:
                semantic_fallback = target.model_copy(update={"runtime_id": None})
                if not any(
                    (
                        semantic_fallback.pid,
                        semantic_fallback.process_name,
                        semantic_fallback.application,
                        semantic_fallback.title,
                        semantic_fallback.class_name,
                        semantic_fallback.query,
                    )
                ):
                    raise
                target = semantic_fallback
        windows = self._enumerate_windows(purpose="target")
        candidates = [window for window in windows if self._matches_structured(window, target)]
        query = (target.query or "").strip().casefold()
        if query:
            candidates = self._query_candidates(candidates, query)
        if not candidates:
            raise WindowsRuntimeError(
                WindowsErrorCode.TARGET_NOT_FOUND,
                "no current window matches the requested target",
            )
        if len(candidates) > 1:
            raise WindowsRuntimeError(
                WindowsErrorCode.AMBIGUOUS_TARGET,
                "multiple current windows match the requested target",
                candidates=candidates[:10],
            )
        return candidates[0]

    def focus_window(self, target: WindowTarget, *, timeout_seconds: float = 2.0) -> WindowsObservation:
        before = self.foreground_window().target
        candidate = self.resolve_window(target)
        hwnd = self._validated_hwnd(candidate)
        if candidate.minimized:
            self._user32.ShowWindow(hwnd, 9)
        self._request_foreground(hwnd)
        after = self._wait_for(candidate, lambda item: item.foreground, timeout_seconds)
        if after is None:
            raise WindowsRuntimeError(
                WindowsErrorCode.FOCUS_FAILED,
                "Windows did not verify the requested window as foreground",
            )
        return self._action_observation("window.focus", candidate, before, after)

    def _request_foreground(self, hwnd: wintypes.HWND) -> None:
        """Request foreground access while respecting Windows' thread-input rules."""
        foreground = wintypes.HWND(self._user32.GetForegroundWindow() or 0)
        current_thread = int(ctypes.windll.kernel32.GetCurrentThreadId())
        foreground_thread = (
            int(self._user32.GetWindowThreadProcessId(foreground, None))
            if foreground
            else 0
        )
        target_thread = int(self._user32.GetWindowThreadProcessId(hwnd, None))
        attached: list[int] = []
        try:
            for thread_id in (foreground_thread, target_thread):
                if thread_id and thread_id != current_thread and thread_id not in attached:
                    if self._user32.AttachThreadInput(current_thread, thread_id, True):
                        attached.append(thread_id)
            self._user32.BringWindowToTop(hwnd)
            self._user32.SetForegroundWindow(hwnd)
        finally:
            for thread_id in reversed(attached):
                self._user32.AttachThreadInput(current_thread, thread_id, False)

    def minimize_window(self, target: WindowTarget, *, timeout_seconds: float = 2.0) -> WindowsObservation:
        return self._show_window("window.minimize", target, 6, lambda item: item.minimized, timeout_seconds)

    def maximize_window(self, target: WindowTarget, *, timeout_seconds: float = 2.0) -> WindowsObservation:
        return self._show_window("window.maximize", target, 3, lambda item: item.maximized, timeout_seconds)

    def restore_window(self, target: WindowTarget, *, timeout_seconds: float = 2.0) -> WindowsObservation:
        candidate = self.resolve_window(target)
        before = candidate
        if not candidate.minimized and not candidate.maximized:
            return self._action_observation("window.restore", candidate, before, candidate)
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        hwnd = self._validated_hwnd(candidate)
        self._user32.ShowWindow(hwnd, 9)
        after = self._wait_for(
            candidate,
            lambda item: not item.minimized,
            max(deadline - time.monotonic(), 0.1),
        )
        if after is not None and after.maximized:
            self._user32.ShowWindow(self._validated_hwnd(after), 9)
            after = self._wait_for(
                after,
                lambda item: not item.minimized and not item.maximized,
                max(deadline - time.monotonic(), 0.1),
            )
        if after is None or after.minimized or after.maximized:
            raise WindowsRuntimeError(
                WindowsErrorCode.TIMEOUT,
                "Windows did not verify restore state",
            )
        return self._action_observation("window.restore", candidate, before, after)

    def launch_application(self, application: str, *, timeout_seconds: float = 5.0) -> WindowsObservation:
        normalized = _APP_ALIASES.get(application.strip().casefold())
        command = _LAUNCH_COMMANDS.get(normalized or "")
        if command is None:
            raise WindowsRuntimeError(
                WindowsErrorCode.OPERATION_NOT_SUPPORTED,
                "application launch is limited to the safe application catalog",
            )
        executable = shutil.which(command[0]) or command[0]
        before_ids = {
            window.runtime_id
            for window in self._enumerate_windows(purpose="target")
        }
        try:
            process = subprocess.Popen(
                [executable, *command[1:]],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise WindowsRuntimeError(
                WindowsErrorCode.LAUNCH_FAILED,
                "Windows could not launch the requested catalog application",
            ) from exc
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        stable_runtime_id = None
        stable_since = None
        while time.monotonic() < deadline:
            matches = []
            for window in self._enumerate_windows(purpose="target"):
                if window.runtime_id not in before_ids and (
                    window.pid == process.pid
                    or (window.process_name or "").casefold() == (normalized or "").casefold()
                ):
                    matches.append(window)
            if len(matches) == 1:
                window = matches[0]
                if stable_runtime_id != window.runtime_id:
                    stable_runtime_id = window.runtime_id
                    stable_since = time.monotonic()
                elif stable_since is not None and time.monotonic() - stable_since >= 1.0:
                    return WindowsObservation(
                        operation="application.launch",
                        verified=True,
                        target=window,
                        after=window,
                    )
            else:
                stable_runtime_id = None
                stable_since = None
            time.sleep(0.05)
        raise WindowsRuntimeError(
            WindowsErrorCode.TIMEOUT,
            "the application process started but no new application window was verified",
        )

    def _show_window(self, operation, target, command, predicate, timeout_seconds):
        candidate = self.resolve_window(target)
        before = candidate
        hwnd = self._validated_hwnd(candidate)
        if predicate(candidate):
            return self._action_observation(operation, candidate, before, candidate)
        self._user32.ShowWindow(hwnd, command)
        after = self._wait_for(candidate, predicate, timeout_seconds)
        if after is None:
            raise WindowsRuntimeError(
                WindowsErrorCode.TIMEOUT,
                f"Windows did not verify {operation.rsplit('.', 1)[-1]} state",
            )
        return self._action_observation(operation, candidate, before, after)

    @staticmethod
    def _action_observation(operation, target, before, after):
        return WindowsObservation(
            operation=operation,
            verified=True,
            target=target,
            before=before,
            after=after,
        )

    def _wait_for(self, candidate, predicate, timeout_seconds):
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        while time.monotonic() < deadline:
            try:
                current = self._resolve_runtime_id(candidate.runtime_id)
            except WindowsRuntimeError as exc:
                if exc.code is WindowsErrorCode.TARGET_STALE:
                    raise
                return None
            if predicate(current):
                return current
            time.sleep(0.05)
        return None

    def _validated_hwnd(self, candidate: WindowRef):
        hwnd = wintypes.HWND(int(candidate.handle, 16))
        if not self._user32.IsWindow(hwnd):
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "the selected window no longer exists")
        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) != candidate.pid:
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "the selected window identity is stale")
        return hwnd

    def _resolve_runtime_id(self, runtime_id: str) -> WindowRef:
        try:
            prefix, pid_text, handle_text = runtime_id.split(":", 2)
            if prefix != "win":
                raise ValueError
            expected_pid = int(pid_text)
            hwnd = int(handle_text, 16)
        except (TypeError, ValueError) as exc:
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "invalid runtime window identity") from exc
        if not self._user32.IsWindow(wintypes.HWND(hwnd)):
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "the selected window no longer exists")
        current = self._window_ref(hwnd)
        if current.pid != expected_pid:
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "the selected window identity is stale")
        classification = classify_window_candidate(self._candidate_state(hwnd))
        if not classification.targetable:
            raise WindowsRuntimeError(
                WindowsErrorCode.TARGET_STALE,
                "the selected window is no longer a safe target",
            )
        return current

    def _raw_window_handles(self) -> list[int]:
        handles: list[int] = []
        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, _lparam):
            handles.append(int(hwnd))
            return True

        if not self._user32.EnumWindows(callback, 0):
            raise WindowsRuntimeError(WindowsErrorCode.WINDOWS_API_ERROR, "Windows could not enumerate top-level windows")
        return handles

    def _enumerate_windows(self, *, purpose: str = "list") -> list[WindowRef]:
        windows: list[WindowRef] = []
        for hwnd in self._raw_window_handles():
            classification = classify_window_candidate(self._candidate_state(hwnd))
            included = (
                classification.targetable
                if purpose == "target"
                else classification.list_eligible
            )
            if included:
                windows.append(self._window_ref(hwnd))
        return windows

    def _candidate_state(self, hwnd: int) -> WindowCandidateState:
        native = wintypes.HWND(hwnd)
        class_buffer = ctypes.create_unicode_buffer(256)
        self._user32.GetClassNameW(native, class_buffer, 256)
        ex_style = int(self._user32.GetWindowLongW(native, -20))
        owner_handle = int(self._user32.GetWindow(native, 4) or 0)  # GW_OWNER
        cloaked_value = False
        try:
            cloaked = wintypes.DWORD()
            result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                native,
                14,  # DWMWA_CLOAKED
                ctypes.byref(cloaked),
                ctypes.sizeof(cloaked),
            )
            cloaked_value = bool(result == 0 and cloaked.value)
        except (AttributeError, OSError):
            pass
        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(native, ctypes.byref(pid))
        return WindowCandidateState(
            exists=bool(self._user32.IsWindow(native)),
            title=self._window_text(hwnd),
            process_name=self._process_name(int(pid.value)),
            class_name=class_buffer.value or None,
            visible=bool(self._user32.IsWindowVisible(native)),
            minimized=bool(self._user32.IsIconic(native)),
            maximized=bool(self._user32.IsZoomed(native)),
            owner_handle=owner_handle,
            tool_window=bool(ex_style & 0x00000080),
            cloaked=cloaked_value,
        )

    def window_diagnostics(self) -> list[dict[str, object]]:
        """Return local debug facts; never exposed as a model capability."""
        diagnostics = []
        for hwnd in self._raw_window_handles():
            state = self._candidate_state(hwnd)
            classification = classify_window_candidate(state)
            diagnostics.append(
                {
                    "handle": hex(hwnd),
                    **asdict(state),
                    **asdict(classification),
                }
            )
        return diagnostics

    def _window_ref(self, hwnd: int) -> WindowRef:
        rect = wintypes.RECT()
        self._user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
        class_buffer = ctypes.create_unicode_buffer(256)
        self._user32.GetClassNameW(wintypes.HWND(hwnd), class_buffer, 256)
        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        process_name = self._process_name(int(pid.value))
        foreground = int(self._user32.GetForegroundWindow() or 0) == hwnd
        handle = hex(hwnd)
        return WindowRef(
            runtime_id=f"win:{int(pid.value)}:{handle}",
            handle=handle,
            pid=int(pid.value),
            process_name=process_name,
            application=self._application_name(process_name),
            title=self._window_text(hwnd),
            class_name=class_buffer.value or None,
            visible=bool(self._user32.IsWindowVisible(wintypes.HWND(hwnd))),
            minimized=bool(self._user32.IsIconic(wintypes.HWND(hwnd))),
            maximized=bool(self._user32.IsZoomed(wintypes.HWND(hwnd))),
            foreground=foreground,
            bounds=WindowBounds(left=rect.left, top=rect.top, right=rect.right, bottom=rect.bottom),
        )

    def _processes(self, visible_counts, *, interactive_only):
        try:
            psutil = importlib.import_module("psutil")
        except ImportError:
            unique = {
                (window.pid, window.process_name or "unknown")
                for window in self._enumerate_windows(purpose="target")
            }
            return [
                ProcessRef(
                    pid=pid,
                    process_name=name,
                    application=self._application_name(name),
                    interactive=True,
                    visible_window_count=visible_counts.get(pid, 0),
                    executable_name=name,
                )
                for pid, name in unique
            ]
        results = []
        for process in psutil.process_iter(("pid", "name", "exe")):
            try:
                pid = int(process.info["pid"])
                name = str(process.info.get("name") or "unknown")
                interactive = visible_counts.get(pid, 0) > 0
                if interactive_only and not interactive:
                    continue
                executable = process.info.get("exe")
                results.append(
                    ProcessRef(
                        pid=pid,
                        process_name=name,
                        application=self._application_name(name),
                        session_id=self._session_id(pid),
                        interactive=interactive,
                        visible_window_count=visible_counts.get(pid, 0),
                        executable_name=os.path.basename(executable) if executable else name,
                    )
                )
            except (OSError, ValueError, psutil.Error):
                continue
        return results

    @staticmethod
    def _matches_structured(window: WindowRef, target: WindowTarget) -> bool:
        pairs = (
            (target.pid, window.pid),
            (target.process_name, window.process_name),
            (target.application, window.application),
            (target.title, window.title),
            (target.class_name, window.class_name),
        )
        for requested, actual in pairs:
            if requested is None:
                continue
            if isinstance(requested, int):
                if requested != actual:
                    return False
            elif actual is None or requested.strip().casefold() != str(actual).strip().casefold():
                return False
        return True

    @staticmethod
    def _query_candidates(windows: list[WindowRef], query: str) -> list[WindowRef]:
        alias = _APP_ALIASES.get(query)
        exact = [
            item
            for item in windows
            if query
            in {
                item.title.casefold(),
                (item.process_name or "").casefold(),
                (item.application or "").casefold(),
                (item.class_name or "").casefold(),
            }
            or (alias and (item.process_name or "").casefold() == alias)
        ]
        if exact:
            return exact
        return [
            item
            for item in windows
            if query in item.title.casefold()
            or query in (item.application or "").casefold()
        ]

    def _window_text(self, hwnd: int) -> str:
        length = self._user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1)
        return buffer.value

    @staticmethod
    def _application_name(process_name: str | None) -> str | None:
        if not process_name:
            return None
        normalized = process_name.casefold()
        for alias, executable in _APP_ALIASES.items():
            if normalized == executable and "." not in alias:
                return alias.title()
        return process_name.rsplit(".", 1)[0]

    @staticmethod
    def _process_name(pid: int) -> str | None:
        try:
            psutil = importlib.import_module("psutil")
            return str(psutil.Process(pid).name())
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _session_id(pid: int) -> int | None:
        try:
            value = wintypes.DWORD()
            if ctypes.windll.kernel32.ProcessIdToSessionId(pid, ctypes.byref(value)):
                return int(value.value)
        except (AttributeError, OSError):
            pass
        return None

    @staticmethod
    def _windows_version() -> tuple[str, str, str]:
        edition = "Windows"
        version = platform.release() or "unknown"
        build = platform.version() or "unknown"
        try:
            winreg = importlib.import_module("winreg")
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            ) as key:
                edition = str(winreg.QueryValueEx(key, "ProductName")[0])
                version = str(winreg.QueryValueEx(key, "DisplayVersion")[0])
                build = str(winreg.QueryValueEx(key, "CurrentBuildNumber")[0])
        except (ImportError, OSError):
            pass
        return edition, version, build


class InMemoryWindowsControlBackend:
    """Deterministic backend for unit tests and in-memory application mode."""

    def __init__(self) -> None:
        self.environment = WindowsEnvironment(
            edition="Windows 11 Pro",
            version="24H2",
            build="26100",
            architecture="AMD64",
            interactive_session=True,
            session_id=1,
        )
        self.windows = [
            WindowRef(runtime_id="win:10:0x1", handle="0x1", pid=10, process_name="editor.exe", application="Editor", title="Notes", foreground=True),
            WindowRef(runtime_id="win:20:0x2", handle="0x2", pid=20, process_name="notepad.exe", application="Notepad", title="Untitled - Notepad"),
        ]
        self.suppressed_runtime_ids: set[str] = set()
        self.launched: list[str] = []

    def inspect_environment(self):
        return WindowsObservation(operation="environment.inspect", verified=True, environment=self.environment)

    def list_processes(self, *, limit=30, interactive_only=False):
        targetable = [w for w in self.windows if w.runtime_id not in self.suppressed_runtime_ids]
        items = [ProcessRef(pid=w.pid, process_name=w.process_name or "unknown", application=w.application, interactive=True, visible_window_count=1, executable_name=w.process_name) for w in targetable]
        return WindowsObservation(operation="process.list", verified=True, processes=items[:limit], count=len(items), truncated=len(items) > limit)

    def list_windows(self, *, limit=30):
        eligible = [w for w in self.windows if w.runtime_id not in self.suppressed_runtime_ids]
        return WindowsObservation(operation="window.list", verified=True, windows=eligible[:limit], count=len(eligible), truncated=len(eligible) > limit)

    def foreground_window(self):
        target = next((item for item in self.windows if item.foreground), None)
        return WindowsObservation(operation="window.foreground", verified=target is not None, target=target)

    def resolve_window(self, target):
        candidates = [
            item
            for item in self.windows
            if item.runtime_id not in self.suppressed_runtime_ids
        ]
        if target.runtime_id:
            candidates = [item for item in candidates if item.runtime_id == target.runtime_id]
            if not candidates:
                semantic_fallback = target.model_copy(update={"runtime_id": None})
                if not any(
                    (
                        semantic_fallback.pid,
                        semantic_fallback.process_name,
                        semantic_fallback.application,
                        semantic_fallback.title,
                        semantic_fallback.class_name,
                        semantic_fallback.query,
                    )
                ):
                    raise WindowsRuntimeError(WindowsErrorCode.TARGET_STALE, "the selected window no longer exists")
                target = semantic_fallback
                candidates = [
                    item
                    for item in self.windows
                    if item.runtime_id not in self.suppressed_runtime_ids
                    and NativeWindowsControlBackend._matches_structured(item, target)
                ]
                if target.query:
                    candidates = NativeWindowsControlBackend._query_candidates(
                        candidates,
                        target.query.casefold(),
                    )
        else:
            candidates = [item for item in candidates if NativeWindowsControlBackend._matches_structured(item, target)]
            if target.query:
                candidates = NativeWindowsControlBackend._query_candidates(candidates, target.query.casefold())
        if not candidates:
            raise WindowsRuntimeError(WindowsErrorCode.TARGET_NOT_FOUND, "no current window matches the requested target")
        if len(candidates) > 1:
            raise WindowsRuntimeError(WindowsErrorCode.AMBIGUOUS_TARGET, "multiple current windows match the requested target", candidates=candidates)
        return candidates[0]

    def focus_window(self, target, *, timeout_seconds=2.0):
        before = self.foreground_window().target
        candidate = self.resolve_window(target)
        self.windows = [item.model_copy(update={"foreground": item.runtime_id == candidate.runtime_id}) for item in self.windows]
        after = self.resolve_window(WindowTarget(runtime_id=candidate.runtime_id))
        return NativeWindowsControlBackend._action_observation("window.focus", candidate, before, after)

    def minimize_window(self, target, *, timeout_seconds=2.0):
        return self._state(target, "window.minimize", minimized=True, maximized=False)

    def maximize_window(self, target, *, timeout_seconds=2.0):
        return self._state(target, "window.maximize", minimized=False, maximized=True)

    def restore_window(self, target, *, timeout_seconds=2.0):
        return self._state(target, "window.restore", minimized=False, maximized=False)

    def launch_application(self, application, *, timeout_seconds=5.0):
        normalized = _APP_ALIASES.get(application.casefold())
        if normalized not in _LAUNCH_COMMANDS:
            raise WindowsRuntimeError(WindowsErrorCode.OPERATION_NOT_SUPPORTED, "application launch is limited to the safe application catalog")
        self.launched.append(normalized)
        pid = max(item.pid for item in self.windows) + 1
        window = WindowRef(runtime_id=f"win:{pid}:{hex(pid)}", handle=hex(pid), pid=pid, process_name=normalized, application=application.title(), title=application.title())
        self.windows.append(window)
        return WindowsObservation(operation="application.launch", verified=True, target=window, after=window)

    def _state(self, target, operation, **updates):
        before = self.resolve_window(target)
        self.windows = [item.model_copy(update=updates) if item.runtime_id == before.runtime_id else item for item in self.windows]
        after = self.resolve_window(WindowTarget(runtime_id=before.runtime_id))
        return NativeWindowsControlBackend._action_observation(operation, before, before, after)
