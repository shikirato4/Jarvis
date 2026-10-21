"""Safe, deterministic validation helpers for generated Python snippets."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ALLOWED_IMPORTS = {
    "collections", "dataclasses", "functools", "itertools", "math", "re", "string", "typing"
}
FORBIDDEN_NAMES = {
    "__import__", "compile", "eval", "exec", "input", "open", "breakpoint", "help", "globals", "locals"
}
FORBIDDEN_MODULES = {"os", "pathlib", "socket", "subprocess", "shutil", "sys", "ctypes", "requests", "urllib"}


def extract_python_solution(response: str) -> str | None:
    """Extract one clearly delimited Python solution without executing prose."""
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        candidates = [block.strip() for block in blocks if re.search(r"\bdef\s+\w+\s*\(", block)]
        return candidates[-1] if len(candidates) == 1 else (candidates[-1] if candidates else None)
    match = re.search(r"(?m)^\s*(?:def|from\s+\w+\s+import|import)\s+", response)
    if not match:
        return None
    candidate = response[match.start():].strip()
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def inspect_safety(code: str) -> tuple[bool, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "SYNTAX_ERROR"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            if any(name in FORBIDDEN_MODULES or name not in ALLOWED_IMPORTS for name in names):
                return False, "UNSAFE_GENERATED_CODE"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORTS:
                return False, "UNSAFE_GENERATED_CODE"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES or isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"system", "popen", "run", "Popen"}:
            return False, "UNSAFE_GENERATED_CODE"
    return True, None


def _runtime_dir() -> Path:
    """Disposable runs directory, created on demand (fresh checkouts lack it)."""
    directory = Path.cwd() / "runtime"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def execute_solution(code: str, tests: str, timeout_seconds: float = 2.0) -> dict[str, object]:
    """Run code plus a fixed harness in a disposable directory."""
    safe, reason = inspect_safety(code)
    if not safe:
        return {"syntax": reason != "SYNTAX_ERROR", "unsafe": reason == "UNSAFE_GENERATED_CODE", "timeout": False, "passed": 0, "total": 0, "exception": reason}
    with tempfile.TemporaryDirectory(prefix="jarvis-coding-", dir=_runtime_dir()) as directory:
        script = Path(directory) / "case.py"
        script.write_text(code + "\n" + tests + "\n", encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=directory,
                capture_output=True, text=True, timeout=timeout_seconds,
                env={"PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return {"syntax": True, "unsafe": False, "timeout": True, "passed": 0, "total": 0, "exception": "EXECUTION_TIMEOUT"}
        output = (completed.stdout or "").strip().splitlines()
        if completed.returncode != 0:
            category = "OTHER:RuntimeError"
            if completed.stderr:
                match = re.search(r"(?:^|\n)([A-Za-z_][A-Za-z0-9_]*Error):", completed.stderr)
                if match:
                    category = match.group(1).upper()
            return {"syntax": True, "unsafe": False, "timeout": False, "passed": 0, "total": 0, "exception": category}
        if not output:
            return {"syntax": True, "unsafe": False, "timeout": False, "passed": 0, "total": 0, "exception": "OTHER:no test report"}
        passed, total = output[-1].split("/", 1)
        return {"syntax": True, "unsafe": False, "timeout": False, "passed": int(passed), "total": int(total), "exception": None}
