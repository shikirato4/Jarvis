from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_OBSOLETE_PREFIXES = (
    "GPT_OSS",
    "GPT-OSS",
    "JARVIS_GPT_OSS",
    "JARVIS_GPT-OSS",
    "OLLAMA",
    "JARVIS_OLLAMA",
)


@dataclass(frozen=True, slots=True)
class LegacyModelConfigurationDiagnostic:
    obsolete_keys: tuple[str, ...]
    ignored: bool = True

    @property
    def detected(self) -> bool:
        return bool(self.obsolete_keys)


def detect_legacy_model_configuration(
    *,
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = _DEFAULT_ENV_FILE,
    emit_log: bool = True,
) -> LegacyModelConfigurationDiagnostic:
    """Report obsolete variable names while never retaining or logging values."""

    names = set((environ or os.environ).keys())
    if env_file is not None:
        names.update(_environment_key_names(Path(env_file)))
    obsolete = tuple(sorted(name for name in names if _is_obsolete_key(name)))
    diagnostic = LegacyModelConfigurationDiagnostic(obsolete_keys=obsolete)
    if emit_log and diagnostic.detected:
        logging.getLogger("jarvis.config").warning(
            "legacy_model_configuration_ignored",
            extra={
                "obsolete_keys": diagnostic.obsolete_keys,
                "obsolete_key_count": len(diagnostic.obsolete_keys),
            },
        )
    return diagnostic


def _environment_key_names(path: Path) -> set[str]:
    names: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.lstrip("\ufeff").strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.casefold().startswith("export "):
                    line = line[7:].lstrip()
                name = line.split("=", 1)[0].strip()
                if name:
                    names.add(name)
    except (OSError, UnicodeError):
        return set()
    return names


def _is_obsolete_key(name: str) -> bool:
    normalized = str(name).strip().upper()
    return any(normalized.startswith(prefix) for prefix in _OBSOLETE_PREFIXES)
