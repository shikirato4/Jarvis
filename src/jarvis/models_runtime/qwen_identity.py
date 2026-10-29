from __future__ import annotations

import re
from dataclasses import dataclass


CANONICAL_QWEN_MODEL = "Qwen3.5-9B-IQ4_XS"


class InvalidQwenIdentityError(ValueError):
    """A configured identifier is not the expected Qwen family."""


def local_model_policy_description() -> str:
    return (
        "Qwen3.5-9B quantized (IQ4_XS GGUF); the configured model identifier and "
        "aliases are matched against the local OpenAI-compatible server"
    )


def _normalize(value: str) -> str:
    return "/".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _family_matches(value: str, family: str) -> bool:
    normalized = str(value).casefold().replace("_", ".")
    family_key = str(family).casefold().replace("_", ".")
    if not family_key.startswith("qwen"):
        return False
    generation = family_key.removeprefix("qwen").strip(" -")
    if not re.fullmatch(r"3\.(?:5|8)", generation):
        return False
    # Require a version separator. This intentionally does not treat Qwen3-8B
    # (a size label) as Qwen 3.8.
    return bool(
        re.search(
            rf"qwen\s*{re.escape(generation)}(?:\b|[-_])",
            normalized,
        )
    ) or bool(re.search(rf"qwen\s*{re.escape(generation)}\b", normalized))


@dataclass(frozen=True, slots=True)
class QwenIdentity:
    configured_model: str = CANONICAL_QWEN_MODEL
    aliases: tuple[str, ...] = ()
    expected_model_family: str = "qwen3.5"

    def __post_init__(self) -> None:
        configured = self.configured_model.strip()
        if configured != CANONICAL_QWEN_MODEL and not _family_matches(
            configured, self.expected_model_family
        ):
            raise InvalidQwenIdentityError(
                "LOCAL model does not identify the expected Qwen model family"
            )
        aliases: list[str] = []
        seen: set[str] = set()
        for alias in self.aliases:
            value = str(alias).strip()
            key = _normalize(value)
            if value and key not in seen:
                seen.add(key)
                aliases.append(value)
        object.__setattr__(self, "configured_model", configured)
        object.__setattr__(self, "aliases", tuple(aliases))

    @property
    def canonical_model(self) -> str:
        return CANONICAL_QWEN_MODEL

    def is_compatible(self, advertised_model: str) -> bool:
        key = _normalize(advertised_model)
        accepted = {_normalize(self.configured_model), *(_normalize(a) for a in self.aliases)}
        return key in accepted

    def select_advertised(self, advertised_models: list[str]) -> str | None:
        for preferred in (self.configured_model, *self.aliases):
            key = _normalize(preferred)
            for model in advertised_models:
                if _normalize(model) == key:
                    return model
        return None
