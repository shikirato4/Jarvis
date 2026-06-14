from __future__ import annotations

import ctypes
import importlib
import time
from ctypes import wintypes
from threading import RLock
from typing import Callable, Protocol

from jarvis.core.errors import UIAutomationError

from .base import MouseButton, PointerPosition, WindowInfo

user32 = ctypes.windll.user32


class DesktopAutomationBackend(Protocol):
    def get_active_window(self) -> WindowInfo | None: ...

    def list_windows(self) -> list[WindowInfo]: ...

    def focus_window(self, target: str, *, timeout_seconds: float) -> WindowInfo: ...

    def close_window(self, target: str | None = None, *, timeout_seconds: float) -> WindowInfo | None: ...

    def move_mouse(self, x: int, y: int, *, duration_seconds: float, relative: bool = False) -> PointerPosition: ...

    def click(self, button: MouseButton, *, double: bool = False) -> None: ...

    def type_text(
        self,
        text: str,
        *,
        interval_seconds: float,
        on_progress: Callable[[], None] | None = None,
    ) -> None: ...

    def hotkey(self, keys: tuple[str, ...]) -> None: ...

    def copy_selection_text(self) -> str: ...


class WindowsDesktopAutomationBackend:
    def __init__(self) -> None:
        self._lock = RLock()

    def get_active_window(self) -> WindowInfo | None:
        handle = user32.GetForegroundWindow()
        if not handle:
            return None
        return self._window_info(handle)

    def list_windows(self) -> list[WindowInfo]:
        windows: list[WindowInfo] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def _enum_proc(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                title = self._window_text(hwnd)
                if title:
                    windows.append(self._window_info(hwnd))
            return True

        user32.EnumWindows(_enum_proc, 0)
        return windows

    def focus_window(self, target: str, *, timeout_seconds: float) -> WindowInfo:
        target_lower = target.casefold()
        candidate = None
        for window in self.list_windows():
            process_name = (window.process_name or "").casefold()
            class_name = (window.class_name or "").casefold()
            if (
                window.handle.casefold() == target_lower
                or target_lower in window.title.casefold()
                or target_lower == process_name
                or target_lower in class_name
            ):
                candidate = window
                break
        if candidate is None:
            raise UIAutomationError(f"window '{target}' was not found")
        hwnd = wintypes.HWND(int(candidate.handle, 16))
        user32.ShowWindow(hwnd, 9)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            active = self.get_active_window()
            if active and active.handle == candidate.handle:
                return active
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.05)
        raise UIAutomationError(f"window '{target}' could not be focused")

    def close_window(self, target: str | None = None, *, timeout_seconds: float) -> WindowInfo | None:
        candidate = self.get_active_window() if not target else None
        if target:
            target_lower = target.casefold()
            for window in self.list_windows():
                process_name = (window.process_name or "").casefold()
                class_name = (window.class_name or "").casefold()
                if (
                    window.handle.casefold() == target_lower
                    or target_lower in window.title.casefold()
                    or target_lower == process_name
                    or target_lower in class_name
                ):
                    candidate = window
                    break
        if candidate is None:
            raise UIAutomationError(f"window '{target or 'active'}' was not found")
        hwnd = wintypes.HWND(int(candidate.handle, 16))
        user32.PostMessageW(hwnd, 0x0010, 0, 0)
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            active = self.get_active_window()
            if active is None or active.handle != candidate.handle:
                return active
            time.sleep(0.05)
        raise UIAutomationError(f"window '{candidate.title}' could not be closed")

    def move_mouse(self, x: int, y: int, *, duration_seconds: float, relative: bool = False) -> PointerPosition:
        with self._lock:
            current = self.get_pointer_position()
            target_x = current.x + x if relative else x
            target_y = current.y + y if relative else y
            steps = max(int(duration_seconds / 0.01), 1) if duration_seconds > 0 else 1
            for step in range(1, steps + 1):
                next_x = int(current.x + ((target_x - current.x) * step / steps))
                next_y = int(current.y + ((target_y - current.y) * step / steps))
                user32.SetCursorPos(next_x, next_y)
                if steps > 1:
                    time.sleep(duration_seconds / steps)
            return self.get_pointer_position()

    def click(self, button: MouseButton, *, double: bool = False) -> None:
        flags = {
            MouseButton.LEFT: (0x0002, 0x0004),
            MouseButton.RIGHT: (0x0008, 0x0010),
            MouseButton.MIDDLE: (0x0020, 0x0040),
        }[button]
        with self._lock:
            self._mouse_event(flags[0])
            self._mouse_event(flags[1])
            if double:
                time.sleep(0.05)
                self._mouse_event(flags[0])
                self._mouse_event(flags[1])

    def type_text(
        self,
        text: str,
        *,
        interval_seconds: float,
        on_progress: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            if self._should_use_clipboard_paste(text=text, interval_seconds=interval_seconds):
                if on_progress is not None:
                    for _ in text:
                        on_progress()
                if self._paste_text_via_clipboard(text):
                    return
            for character in text:
                if on_progress is not None:
                    on_progress()
                self._unicode_key(character)
                if interval_seconds > 0:
                    time.sleep(interval_seconds)

    def hotkey(self, keys: tuple[str, ...]) -> None:
        virtual_keys = [self._virtual_key(key) for key in keys]
        with self._lock:
            for key in virtual_keys:
                self._key_event(key, key_up=False)
            for key in reversed(virtual_keys):
                self._key_event(key, key_up=True)

    def copy_selection_text(self) -> str:
        with self._lock:
            previous_text = self._clipboard_text()
            if previous_text is None:
                return ""
            self.hotkey(("ctrl", "c"))
            time.sleep(0.05)
            selected_text = self._clipboard_text()
            self._set_clipboard_text(previous_text)
            return str(selected_text or "")

    def get_pointer_position(self) -> PointerPosition:
        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        return PointerPosition(x=int(point.x), y=int(point.y))

    def _window_info(self, hwnd) -> WindowInfo:
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process_name = self._process_name(int(process_id.value or 0))
        return WindowInfo(
            handle=hex(int(hwnd)),
            title=self._window_text(hwnd),
            class_name=class_buffer.value or None,
            process_id=int(process_id.value or 0),
            process_name=process_name,
            rect={
                "left": int(rect.left),
                "top": int(rect.top),
                "right": int(rect.right),
                "bottom": int(rect.bottom),
            },
        )

    @staticmethod
    def _window_text(hwnd) -> str:
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value

    @staticmethod
    def _mouse_event(flag: int) -> None:
        user32.mouse_event(flag, 0, 0, 0, 0)

    @staticmethod
    def _unicode_key(character: str) -> None:
        value = ord(character)
        WindowsDesktopAutomationBackend._keybd_unicode(value, key_up=False)
        WindowsDesktopAutomationBackend._keybd_unicode(value, key_up=True)

    @staticmethod
    def _keybd_unicode(scan_code: int, *, key_up: bool) -> None:
        user32.keybd_event(0, scan_code, 0x0004 | (0x0002 if key_up else 0), 0)

    @staticmethod
    def _key_event(virtual_key: int, *, key_up: bool) -> None:
        user32.keybd_event(virtual_key, 0, 0x0002 if key_up else 0, 0)

    @staticmethod
    def _virtual_key(key: str) -> int:
        normalized = key.casefold()
        mapping = {
            "ctrl": 0x11,
            "control": 0x11,
            "shift": 0x10,
            "alt": 0x12,
            "win": 0x5B,
            "enter": 0x0D,
            "tab": 0x09,
            "esc": 0x1B,
            "escape": 0x1B,
            "space": 0x20,
            "backspace": 0x08,
            "delete": 0x2E,
            "up": 0x26,
            "down": 0x28,
            "left": 0x25,
            "right": 0x27,
        }
        if normalized in mapping:
            return mapping[normalized]
        if len(normalized) == 1:
            code = user32.VkKeyScanW(ord(normalized))
            return int(code & 0xFF)
        raise UIAutomationError(f"unsupported key '{key}'")

    @staticmethod
    def _process_name(process_id: int) -> str | None:
        if process_id <= 0:
            return None
        try:
            psutil = importlib.import_module("psutil")
            process = psutil.Process(process_id)
            return process.name()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _should_use_clipboard_paste(*, text: str, interval_seconds: float) -> bool:
        return bool(text) and (len(text) >= 48 or "\n" in text or interval_seconds <= 0.004)

    def _paste_text_via_clipboard(self, text: str) -> bool:
        clipboard_text = self._clipboard_text()
        if clipboard_text is None:
            return False
        previous_text = clipboard_text
        if not self._set_clipboard_text(text):
            return False
        try:
            self.hotkey(("ctrl", "v"))
        finally:
            self._set_clipboard_text(previous_text)
        return True

    @staticmethod
    def _clipboard_text() -> str | None:
        try:
            win32clipboard = importlib.import_module("win32clipboard")
            win32con = importlib.import_module("win32con")
        except ImportError:
            return None
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                return str(win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT))
            return ""
        except Exception:  # noqa: BLE001
            return None
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _set_clipboard_text(text: str) -> bool:
        try:
            win32clipboard = importlib.import_module("win32clipboard")
            win32con = importlib.import_module("win32con")
        except ImportError:
            return False
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:  # noqa: BLE001
                pass


class InMemoryDesktopAutomationBackend:
    def __init__(self) -> None:
        self._active_window = WindowInfo(handle="0x1", title="Editor", class_name="FakeEditor", process_id=1, process_name="editor.exe")
        self._known_windows = [
            self._active_window,
            WindowInfo(handle="0x2", title="Word", class_name="FakeWord", process_id=2, process_name="WINWORD.EXE"),
            WindowInfo(handle="0x3", title="Notepad", class_name="FakeNotepad", process_id=3, process_name="notepad.exe"),
            WindowInfo(handle="0x4", title="VSCode", class_name="FakeVSCode", process_id=4, process_name="Code.exe"),
            WindowInfo(handle="0x5", title="Chrome", class_name="FakeChrome", process_id=5, process_name="chrome.exe"),
            WindowInfo(handle="0x6", title="Opera GX", class_name="FakeOpera", process_id=6, process_name="launcher.exe"),
            WindowInfo(handle="0x7", title="Calculadora", class_name="FakeCalc", process_id=7, process_name="calc.exe"),
            WindowInfo(handle="0x8", title="Explorador de archivos", class_name="FakeExplorer", process_id=8, process_name="explorer.exe"),
        ]
        self.pointer = PointerPosition(x=0, y=0)
        self.typed_text = ""
        self.hotkeys: list[tuple[str, ...]] = []
        self.clicks: list[tuple[str, bool]] = []

    def get_active_window(self) -> WindowInfo | None:
        return self._active_window

    def list_windows(self) -> list[WindowInfo]:
        windows = [self._active_window]
        windows.extend(window for window in self._known_windows if window.handle != self._active_window.handle)
        return windows

    def focus_window(self, target: str, *, timeout_seconds: float) -> WindowInfo:
        for window in self.list_windows():
            lowered = target.casefold()
            if (
                lowered in window.title.casefold()
                or lowered == window.handle.casefold()
                or lowered == (window.process_name or "").casefold()
                or lowered in (window.class_name or "").casefold()
            ):
                self._active_window = window
                return window
        raise UIAutomationError("window not found")

    def close_window(self, target: str | None = None, *, timeout_seconds: float) -> WindowInfo | None:
        candidate = self._active_window if not target else None
        if target:
            lowered = target.casefold()
            for window in self.list_windows():
                if (
                    lowered in window.title.casefold()
                    or lowered == window.handle.casefold()
                    or lowered == (window.process_name or "").casefold()
                    or lowered in (window.class_name or "").casefold()
                ):
                    candidate = window
                    break
        if candidate is None:
            raise UIAutomationError("window not found")
        self._known_windows = [window for window in self._known_windows if window.handle != candidate.handle]
        if self._active_window.handle == candidate.handle:
            self._active_window = self._known_windows[0] if self._known_windows else WindowInfo(handle="0x0", title="Desktop")
        return self._active_window

    def move_mouse(self, x: int, y: int, *, duration_seconds: float, relative: bool = False) -> PointerPosition:
        if relative:
            self.pointer = PointerPosition(x=self.pointer.x + x, y=self.pointer.y + y)
        else:
            self.pointer = PointerPosition(x=x, y=y)
        return self.pointer

    def click(self, button: MouseButton, *, double: bool = False) -> None:
        self.clicks.append((button.value, double))

    def type_text(self, text: str, *, interval_seconds: float, on_progress: Callable[[], None] | None = None) -> None:
        for char in text:
            if on_progress is not None:
                on_progress()
            self.typed_text += char
            if interval_seconds > 0:
                time.sleep(min(interval_seconds, 0.0001))

    def hotkey(self, keys: tuple[str, ...]) -> None:
        self.hotkeys.append(keys)

    def copy_selection_text(self) -> str:
        return self.typed_text[-4000:]
