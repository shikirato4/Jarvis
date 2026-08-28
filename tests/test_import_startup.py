from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def test_package_import_is_lazy_and_does_not_load_full_runtime() -> None:
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys,jarvis; print(json.dumps({'bootstrap': 'jarvis.bootstrap' in sys.modules}))",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    elapsed = perf_counter() - started
    payload = json.loads(completed.stdout)

    assert payload["bootstrap"] is False
    assert elapsed < 5.0


def test_cli_help_surface_does_not_import_bootstrap_or_heavy_science_runtime() -> None:
    started = perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,jarvis.cli; "
                "print(json.dumps({'bootstrap': 'jarvis.bootstrap' in sys.modules, "
                "'science_service': 'jarvis.science_runtime.service' in sys.modules}))"
            ),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    elapsed = perf_counter() - started
    payload = json.loads(completed.stdout)

    assert payload == {"bootstrap": False, "science_service": False}
    assert elapsed < 5.0


def test_desktop_entrypoint_import_is_lightweight_before_splash_is_created() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,jarvis.desktop; import jarvis.desktop_runtime.window; "
                "print(json.dumps({'bootstrap': 'jarvis.bootstrap' in sys.modules, "
                "'desktop_service': 'jarvis.desktop_runtime.service' in sys.modules}))"
            ),
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert json.loads(completed.stdout) == {
        "bootstrap": False,
        "desktop_service": False,
    }
