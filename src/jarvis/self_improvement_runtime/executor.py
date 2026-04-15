from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from .models import SelfImprovementCommandResult, SelfImprovementTestResult

_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+)\s+passed|(?P<failed>\d+)\s+failed|(?P<errors>\d+)\s+error(?:s)?|(?P<skipped>\d+)\s+skipped",
    re.IGNORECASE,
)


class SelfImprovementExecutor:
    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("jarvis.self_improvement.executor")

    def run_tests(self, *, cwd: Path, test_targets: tuple[str, ...]) -> SelfImprovementTestResult:
        command = [sys.executable, "-m", "pytest"]
        if test_targets:
            command.extend(test_targets)
        completed, duration_ms = self._run_command(command, cwd=cwd)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        summary_source = "\n".join(line for line in (stdout + "\n" + stderr).splitlines() if "passed" in line or "failed" in line or "error" in line)
        counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        for match in _PYTEST_SUMMARY_RE.finditer(summary_source):
            for key in counts:
                value = match.group(key)
                if value is not None:
                    counts[key] += int(value)
        summary = next((line.strip() for line in reversed((stdout + "\n" + stderr).splitlines()) if "passed" in line or "failed" in line or "error" in line), "")
        return SelfImprovementTestResult(
            command=tuple(command),
            cwd=str(cwd),
            exit_code=completed.returncode,
            passed=counts["passed"],
            failed=counts["failed"],
            errors=counts["errors"],
            skipped=counts["skipped"],
            summary=summary,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
        )

    def run_compileall(self, *, cwd: Path, changed_files: tuple[str, ...]) -> SelfImprovementCommandResult:
        command = [sys.executable, "-m", "compileall", "-q", *changed_files]
        completed, _duration_ms = self._run_command(command, cwd=cwd)
        stderr = completed.stderr or ""
        stdout = completed.stdout or ""
        summary = "compileall ok" if completed.returncode == 0 else "compileall detectó errores"
        return SelfImprovementCommandResult(
            ok=completed.returncode == 0,
            command=tuple(command),
            cwd=str(cwd),
            exit_code=completed.returncode,
            summary=summary,
            stdout=stdout,
            stderr=stderr,
        )

    def run_import_checks(self, *, cwd: Path, changed_files: tuple[str, ...]) -> SelfImprovementCommandResult:
        script = (
            "import importlib.util, pathlib, sys\n"
            "root = pathlib.Path(sys.argv[1]).resolve()\n"
            "failures = []\n"
            "for index, raw in enumerate(sys.argv[2:]):\n"
            "    target = (root / raw).resolve()\n"
            "    try:\n"
            "        spec = importlib.util.spec_from_file_location(f'_jarvis_self_improvement_{index}', target)\n"
            "        if spec is None or spec.loader is None:\n"
            "            raise RuntimeError('spec unavailable')\n"
            "        module = importlib.util.module_from_spec(spec)\n"
            "        spec.loader.exec_module(module)\n"
            "    except Exception as exc:\n"
            "        failures.append(f'{raw}: {exc.__class__.__name__}: {exc}')\n"
            "if failures:\n"
            "    print('\\n'.join(failures), file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "print(f'Imported {len(sys.argv) - 2} file(s) successfully.')\n"
        )
        command = [sys.executable, "-c", script, str(cwd), *changed_files]
        completed, _duration_ms = self._run_command(command, cwd=cwd)
        return SelfImprovementCommandResult(
            ok=completed.returncode == 0,
            command=tuple(command),
            cwd=str(cwd),
            exit_code=completed.returncode,
            summary=(completed.stdout or completed.stderr or "").strip() or "import validation completed",
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    @staticmethod
    def _run_command(command: list[str], *, cwd: Path) -> tuple[subprocess.CompletedProcess[str], float]:
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
        duration_ms = (time.perf_counter() - started) * 1000
        return completed, duration_ms
