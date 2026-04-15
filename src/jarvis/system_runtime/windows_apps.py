from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WindowsAppDescriptor:
    canonical_id: str
    display_name: str
    aliases: tuple[str, ...]
    executables: tuple[str, ...]
    app_paths: tuple[str, ...]
    install_relative_paths: tuple[str, ...]
    start_menu_aliases: tuple[str, ...] = ()


WINDOWS_APP_CATALOG: tuple[WindowsAppDescriptor, ...] = (
    WindowsAppDescriptor(
        canonical_id="winword",
        display_name="Microsoft Word",
        aliases=("word", "microsoft word", "winword", "word de microsoft"),
        executables=("WINWORD.EXE",),
        app_paths=("WINWORD.EXE",),
        install_relative_paths=(
            "Microsoft Office/root/Office16/WINWORD.EXE",
            "Microsoft Office/root/Office15/WINWORD.EXE",
            "Microsoft Office/root/Office14/WINWORD.EXE",
            "Microsoft Office/Office16/WINWORD.EXE",
            "Microsoft Office/Office15/WINWORD.EXE",
            "Microsoft Office/Office14/WINWORD.EXE",
        ),
        start_menu_aliases=("word", "microsoft word"),
    ),
    WindowsAppDescriptor(
        canonical_id="notepad",
        display_name="Notepad",
        aliases=("notepad", "bloc de notas", "block de notas", "editor de texto"),
        executables=("notepad.exe",),
        app_paths=("notepad.exe",),
        install_relative_paths=("Windows/System32/notepad.exe",),
        start_menu_aliases=("notepad", "bloc de notas"),
    ),
    WindowsAppDescriptor(
        canonical_id="code",
        display_name="Visual Studio Code",
        aliases=("vscode", "vs code", "visual studio code", "code"),
        executables=("Code.exe", "code.exe"),
        app_paths=("Code.exe", "code.exe"),
        install_relative_paths=(
            "Microsoft VS Code/Code.exe",
            "Programs/Microsoft VS Code/Code.exe",
        ),
        start_menu_aliases=("visual studio code", "vscode", "vs code"),
    ),
    WindowsAppDescriptor(
        canonical_id="chrome",
        display_name="Google Chrome",
        aliases=("chrome", "google chrome"),
        executables=("chrome.exe",),
        app_paths=("chrome.exe",),
        install_relative_paths=("Google/Chrome/Application/chrome.exe",),
        start_menu_aliases=("chrome", "google chrome"),
    ),
    WindowsAppDescriptor(
        canonical_id="opera",
        display_name="Opera",
        aliases=("opera", "opera browser", "opera gx"),
        executables=("opera.exe", "launcher.exe"),
        app_paths=("opera.exe", "launcher.exe"),
        install_relative_paths=(
            "Opera/launcher.exe",
            "Programs/Opera/launcher.exe",
            "Programs/Opera GX/launcher.exe",
        ),
        start_menu_aliases=("opera", "opera gx"),
    ),
    WindowsAppDescriptor(
        canonical_id="calc",
        display_name="Calculadora",
        aliases=("calculadora", "calculator", "calc"),
        executables=("calc.exe",),
        app_paths=("calc.exe",),
        install_relative_paths=("Windows/System32/calc.exe",),
        start_menu_aliases=("calculadora", "calculator"),
    ),
    WindowsAppDescriptor(
        canonical_id="explorer",
        display_name="Explorador de archivos",
        aliases=("explorador", "explorer", "file explorer", "explorador de archivos"),
        executables=("explorer.exe",),
        app_paths=("explorer.exe",),
        install_relative_paths=("Windows/explorer.exe",),
        start_menu_aliases=("explorador", "explorer", "file explorer"),
    ),
)

TRUSTED_WINDOWS_APP_IDS = frozenset(item.canonical_id for item in WINDOWS_APP_CATALOG)

TRUSTED_WINDOWS_EXECUTABLES = frozenset(
    executable.casefold()
    for item in WINDOWS_APP_CATALOG
    for executable in item.executables
)


def normalize_windows_app_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    lowered = stripped.casefold()
    lowered = lowered.replace("&", " ")
    lowered = re.sub(r"[^\w\s.-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered.removesuffix(".exe")


def catalog_descriptor_for_query(query: str) -> WindowsAppDescriptor | None:
    normalized = normalize_windows_app_name(query)
    if not normalized:
        return None
    for descriptor in WINDOWS_APP_CATALOG:
        alias_set = {normalize_windows_app_name(item) for item in descriptor.aliases}
        alias_set.add(normalize_windows_app_name(descriptor.display_name))
        alias_set.update(normalize_windows_app_name(item) for item in descriptor.executables)
        if normalized in alias_set:
            return descriptor
    return None


def app_match_score(query: str, *, display_name: str, executable_name: str | None = None, aliases: tuple[str, ...] = ()) -> float:
    normalized = normalize_windows_app_name(query)
    display = normalize_windows_app_name(display_name)
    executable = normalize_windows_app_name(executable_name or "")
    alias_values = {normalize_windows_app_name(item) for item in aliases}
    if normalized == display:
        return 1.0
    if normalized == executable and executable:
        return 0.98
    if normalized in alias_values:
        return 0.97
    if display.startswith(normalized):
        return 0.93
    if normalized and normalized in display:
        return 0.9
    if executable and normalized and normalized in executable:
        return 0.88
    return 0.0


def is_probable_shortcut(path: Path) -> bool:
    return path.suffix.casefold() == ".lnk"
