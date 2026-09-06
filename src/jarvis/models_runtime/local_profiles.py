from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from jarvis.config import Settings


BASELINE_PROFILE_ID = "qwen38-27b-iq4-xs"
QWEN35_PROFILE_ID = "qwen35-9b-iq4-xs"
_QWEN35_MODEL = "Qwen3.5-9B-IQ4_XS"
_QWEN35_PATH = Path("models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf")
_QWEN38_MODEL = "Qwen3.8-27B-IQ4_XS"
_QWEN38_PATH = Path("models/llm/qwen3.8-27b/Qwen3.8-27B-IQ4_XS.gguf")
_RESERVED_FLAGS = {
    "--model",
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
    # the old profile independently selectable even when only its profile id
    # is overridden, while preserving explicit custom baseline settings.
    promoted_defaults = (
        settings.local_llm_model == _QWEN35_MODEL
        and Path(settings.local_runtime_model_path) == _QWEN35_PATH
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
        path = Path(settings.local_runtime_model_path)
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
        llama_cpp_args=(),
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
            vision_capable=False,
            audio_capable=False,
            video_capable=False,
            native_transport_modalities=(),
        ),
    )


def resolve_local_model_profile(
    settings: Settings,
    profiles: Iterable[LocalModelProfile] = (),
) -> LocalModelProfile:
    baseline = baseline_local_model_profile(settings)
    available = {baseline.id: baseline}
    available.update(
        {profile.id: profile for profile in registered_local_model_profiles(settings)}
    )
    available.update({profile.id: profile for profile in profiles})
    selected = str(getattr(settings, "local_runtime_profile_id", BASELINE_PROFILE_ID)).strip()
    try:
        return available[selected]
    except KeyError as exc:
        raise LocalModelProfileConfigurationError(
            f"local runtime profile does not exist: {selected}"
        ) from exc
