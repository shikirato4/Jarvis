from __future__ import annotations

import importlib
import logging
import logging.config
from pathlib import Path


def test_standard_logging_module_is_not_shadowed() -> None:
    module = importlib.import_module("jarvis.jarvis_logging")
    assert hasattr(module, "configure_logging")
    assert hasattr(logging, "getLogger")
    assert hasattr(logging.config, "dictConfig")
    assert not Path("src/jarvis/logging.py").exists()
