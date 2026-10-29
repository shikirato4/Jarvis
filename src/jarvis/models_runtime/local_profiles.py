from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Iterable

from jarvis.models_runtime.qwen_identity import (
    CANONICAL_QWEN_MODEL,
    InvalidQwenIdentityError,
    QwenIdentity,
)

if TYPE_CHECKING:
    from jarvis.config import Settings


logger = logging.getLogger(__name__)


BASELINE_PROFILE_ID = "qwen38-27b-iq4-xs"
QWEN35_PROFILE_ID = "qwen35-9b-iq4-xs"
CONFIGURED_PROFILE_ID = "configured-local-gguf"
_QWEN35_MODEL = "Qwen3.5-9B-IQ4_XS"
_QWEN35_PATH = Path("models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf")
_QWEN38_MODEL = "Qwen3.8-27B-IQ4_XS"
_QWEN38_PATH = Path("models/llm/qwen3.8-27b/Qwen3.8-27B-IQ4_XS.gguf")
_RESERVED_FLAGS = {
    "--model",
    "--mmproj",
    "--alias",
    "--ctx-size",
    "--parallel",
    "--host",
    "--port",
}


class LocalModelProfileConfigurationError(ValueError):
    """A local physical model profile cannot be used safely."""


@dataclass(frozen=True, slots=True)
class LocalModelProfile:
    id: str
    display_name: str
    model_alias: str
    gguf_path: Path
    expected_model_family: str
    context_size: int
    parallel: int
    mmproj_path: Path | None = None
    model_aliases: tuple[str, ...] = ()
    reasoning_modes: tuple[str, ...] = ("off", "low", "medium", "xhigh", "auto")
    reasoning_adapter: str = "qwen_enable_thinking"
    vision_capable: bool = True
    audio_capable: bool = True
    video_capable: bool = True
    native_transport_modalities: tuple[str, ...] = ("image",)
    tool_adapter_requirements: tuple[str, ...] = (
        "dotted_action_names",
        "compact_large_registry",
    )
    llama_cpp_args: tuple[tuple[str, str | None], ...] = ()
    # Adapter LoRA (GGUF) del fine-tune de JARVIS: se aplica en el servidor,
    # separado del modelo base, para poder prenderlo y apagarlo.
    lora_path: Path | None = None

    def __post_init__(self) -> None:
        values = {
            "id": self.id,
            "display_name": self.display_name,
            "model_alias": self.model_alias,
            "expected_model_family": self.expected_model_family,
        }
        if any(not str(value).strip() for value in values.values()):
            raise LocalModelProfileConfigurationError("profile identity fields are required")
        if self.context_size <= 0 or self.parallel <= 0:
            raise LocalModelProfileConfigurationError(
                "profile context_size and parallel must be positive"
            )
        if self.gguf_path.suffix.casefold() != ".gguf":
            raise LocalModelProfileConfigurationError("profile gguf_path must end in .gguf")
        if self.mmproj_path is not None and Path(self.mmproj_path).suffix.casefold() != ".gguf":
            raise LocalModelProfileConfigurationError("profile mmproj_path must end in .gguf")
        if self.lora_path is not None and Path(self.lora_path).suffix.casefold() != ".gguf":
            raise LocalModelProfileConfigurationError("profile lora_path must end in .gguf")
        aliases = tuple(dict.fromkeys(str(alias).strip() for alias in self.model_aliases if str(alias).strip()))
        object.__setattr__(self, "model_aliases", aliases)
        checked_args: list[tuple[str, str | None]] = []
        for raw_flag, raw_value in self.llama_cpp_args:
            flag = str(raw_flag).strip()
            if not flag.startswith("--") or flag in _RESERVED_FLAGS:
                raise LocalModelProfileConfigurationError(
                    f"llama_cpp_args contains reserved flag {flag!r}"
                )
            value = None if raw_value is None else str(raw_value)
            if "\x00" in flag or (value is not None and "\x00" in value):
                raise LocalModelProfileConfigurationError("llama_cpp_args contains a NUL")
            checked_args.append((flag, value))
        object.__setattr__(self, "llama_cpp_args", tuple(checked_args))


def baseline_local_model_profile(settings: Settings) -> LocalModelProfile:
    """Build the retained Qwen3.8 rollback profile from settings when customized."""

    # After promotion, the ordinary Settings defaults describe Qwen3.5. Keep
    # the old profile independently selectable when only its profile id is
    # overridden, while explicit custom model settings must win over the
    # shipped rollback identifiers.
    configured_path = Path(settings.local_runtime_model_path)
    shared_directory_mode = configured_path.suffix.casefold() != ".gguf"
    promoted_defaults = shared_directory_mode and (
        settings.local_llm_model == _QWEN35_MODEL
        and not settings.local_llm_model_aliases
        and settings.local_runtime_context_size == 8192
        and settings.local_runtime_parallel == 1
    )
    if promoted_defaults:
        model = _QWEN38_MODEL
        aliases = (_QWEN38_MODEL,)
        path = _QWEN38_PATH
        context = 4096
        parallel = 1
    else:
        model = settings.local_llm_model
        aliases = tuple(settings.local_llm_model_aliases)
        # The baseline is a selectable identity: it must build even when the
        # shared directory has no artifact yet. Only an explicit .gguf path
        # overrides the shipped rollback artifact.
        path = (
            configured_path
            if configured_path.suffix.casefold() == ".gguf"
            else _QWEN38_PATH
        )
        context = settings.local_runtime_context_size
        parallel = settings.local_runtime_parallel

    return LocalModelProfile(
        id=BASELINE_PROFILE_ID,
        display_name=model,
        model_alias=model,
        model_aliases=aliases,
        gguf_path=path,
        expected_model_family="qwen3.8",
        context_size=context,
        parallel=parallel,
        mmproj_path=_discover_mmproj(path, settings.local_runtime_mmproj_path),
        vision_capable=_discover_mmproj(path, settings.local_runtime_mmproj_path) is not None,
        native_transport_modalities=(
            ("image",) if _discover_mmproj(path, settings.local_runtime_mmproj_path) is not None else ()
        ),
        llama_cpp_args=(),
        lora_path=settings.local_runtime_lora_path,
    )


def registered_local_model_profiles(settings: Settings) -> tuple[LocalModelProfile, ...]:
    """Return shipped candidates without changing the baseline selection."""

    return (
        LocalModelProfile(
            id=QWEN35_PROFILE_ID,
            display_name=_QWEN35_MODEL,
            model_alias=_QWEN35_MODEL,
            gguf_path=_QWEN35_PATH,
            expected_model_family="qwen3.5",
            context_size=settings.local_runtime_context_override or 8192,
            parallel=1,
            reasoning_adapter="qwen_enable_thinking_binary",
            mmproj_path=_discover_mmproj(_QWEN35_PATH, settings.local_runtime_mmproj_path),
            vision_capable=_discover_mmproj(_QWEN35_PATH, settings.local_runtime_mmproj_path) is not None,
            audio_capable=False,
            video_capable=False,
            native_transport_modalities=(
                ("image",)
                if _discover_mmproj(_QWEN35_PATH, settings.local_runtime_mmproj_path) is not None
                else ()
            ),
            lora_path=settings.local_runtime_lora_path,
        ),
    )


def _discover_mmproj(model_path: Path, configured: Path | None) -> Path | None:
    """Resolve an explicitly configured or unambiguous sidecar projector."""
    if configured is not None:
        return Path(configured)
    model_path = Path(model_path)
    exact = model_path.parent / f"mmproj-{model_path.stem}.gguf"
    generic = model_path.parent / "mmproj.gguf"
    for candidate in (exact, generic):
        if candidate.is_file():
            return candidate
    sidecars = sorted(
        candidate
        for candidate in model_path.parent.glob("*.gguf")
        if candidate.name.casefold().startswith("mmproj-")
    )
    if len(sidecars) == 1:
        return sidecars[0]
    return None


def _configured_folder(settings: Settings) -> Path:
    """Resolve the configured artifact path to the folder used for discovery."""
    configured = Path(settings.local_runtime_model_path).expanduser()
    workspace_root = Path(settings.resolved_workspace_root)
    folder = configured if configured.is_absolute() else workspace_root / configured
    return folder.resolve(strict=False)


def _is_base_model_gguf(candidate: Path) -> bool:
    """Un sidecar mmproj o un adapter LoRA no son modelos base elegibles."""

    nombre = candidate.name.casefold()
    if nombre.startswith("mmproj-") or nombre == "mmproj.gguf":
        return False
    return "lora" not in nombre and "adapter" not in nombre


def _configured_model_path(settings: Settings) -> Path:
    """Accept an explicit GGUF or discover the sole model in the shared folder."""
    configured = Path(settings.local_runtime_model_path).expanduser()
    folder = _configured_folder(settings)
    if not folder.is_dir():
        return configured

    candidates = sorted(
        candidate for candidate in folder.glob("*.gguf") if _is_base_model_gguf(candidate)
    )
    if not candidates:
        # Layout real del disco: models/llm/<modelo>/<modelo>.gguf. Si la
        # carpeta compartida solo contiene subcarpetas por modelo, se busca
        # un nivel abajo antes de declarar la configuracion rota, para no
        # romper cuando el repo ya tiene descargas organizadas por modelo.
        candidates = sorted(
            candidate
            for candidate in folder.glob("*/*.gguf")
            if _is_base_model_gguf(candidate)
        )
    if not candidates:
        raise LocalModelProfileConfigurationError(
            f"No hay un modelo GGUF en la carpeta local configurada: {folder}"
        )
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise LocalModelProfileConfigurationError(
            f"Hay varios modelos GGUF en {folder}; deja solo el modelo que quieras usar. "
            f"Encontrados: {names}"
        )
    return candidates[0]


def _qwen_family_alias(value: str) -> str:
    """Keep a configured alias only when it identifies a supported Qwen family.

    A placeholder identifier from a stale environment (for example
    ``local-model``) must never become the provider identity: the canonical
    Qwen model is the safe fallback.
    """
    for family in ("qwen3.5", "qwen3.8"):
        try:
            QwenIdentity(value, expected_model_family=family)
            return str(value)
        except InvalidQwenIdentityError:
            continue
    return CANONICAL_QWEN_MODEL


def configured_local_model_profile(settings: Settings) -> LocalModelProfile:
    """Build a model-neutral profile from the configured artifact and alias."""
    path = _configured_model_path(settings)
    mmproj = _discover_mmproj(path, settings.local_runtime_mmproj_path)
    model = _qwen_family_alias(settings.local_llm_model)
    return LocalModelProfile(
        id=CONFIGURED_PROFILE_ID,
        display_name=model,
        model_alias=model,
        model_aliases=tuple(settings.local_llm_model_aliases),
        gguf_path=path,
        expected_model_family="configured",
        context_size=settings.local_runtime_context_override or settings.local_runtime_context_size,
        parallel=settings.local_runtime_parallel,
        mmproj_path=mmproj,
        reasoning_adapter="none",
        vision_capable=mmproj is not None,
        audio_capable=False,
        video_capable=False,
        native_transport_modalities=("image",) if mmproj is not None else (),
        lora_path=settings.local_runtime_lora_path,
    )


def resolve_local_model_profile(
    settings: Settings,
    profiles: Iterable[LocalModelProfile] = (),
) -> LocalModelProfile:
    """Resolve the selected profile without requiring a local GGUF artifact.

    The configured profile is the only one whose construction needs a real
    artifact, so it is built lazily: only when it is the selected profile and
    no caller supplied it. A configured folder that exists but does not hold
    exactly one GGUF is a misconfiguration and fails with its own readable
    message; a configured path that does not exist at all is the shipped
    ``models/llm`` placeholder of a fresh checkout, and degrades to the
    registered default profile with a warning.
    """
    baseline = baseline_local_model_profile(settings)
    registered = registered_local_model_profiles(settings)
    available = {baseline.id: baseline}
    available.update({profile.id: profile for profile in registered})
    available.update({profile.id: profile for profile in profiles})
    selected = str(getattr(settings, "local_runtime_profile_id", BASELINE_PROFILE_ID)).strip()
    if selected != CONFIGURED_PROFILE_ID or selected in available:
        try:
            return available[selected]
        except KeyError as exc:
            raise LocalModelProfileConfigurationError(
                f"local runtime profile does not exist: {selected}"
            ) from exc
    try:
        return configured_local_model_profile(settings)
    except LocalModelProfileConfigurationError:
        if _configured_folder(settings).is_dir():
            raise
        logger.warning(
            "configured local model artifact not found under %s; using profile %s",
            _configured_folder(settings),
            registered[0].id,
        )
        return registered[0]
