from __future__ import annotations

import re
from dataclasses import dataclass


CANONICAL_NEMOTRON_MODEL = (
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
)
_REQUIRED_FAMILY_TOKENS = frozenset(
    {"nemotron", "3", "nano", "omni", "30b", "a3b", "reasoning"}
)


class InvalidNemotronIdentityError(ValueError):
    """A configured identifier is not the supported Nemotron model family."""


def local_model_policy_description() -> str:
    return (
        "Nemotron 3 Nano Omni 30B A3B Reasoning only; "
        "configured aliases for NVFP4/GGUF via NVIDIA runtime, vLLM, "
        "LM Studio or llama.cpp"
    )


@dataclass(frozen=True, slots=True)
class NemotronIdentity:
    """Canonical model-family identity independent of serving backend names."""

    configured_model: str = CANONICAL_NEMOTRON_MODEL
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        configured = self.configured_model.strip()
        if not _is_supported_family_identifier(configured):
            raise InvalidNemotronIdentityError(
                "LOCAL model must identify Nemotron 3 Nano Omni 30B A3B Reasoning"
            )
        normalized_aliases: list[str] = []
        seen: set[str] = set()
        for alias in self.aliases:
            value = str(alias).strip()
            normalized = _normalize_identifier(value)
            if not value or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_aliases.append(value)
        object.__setattr__(self, "configured_model", configured)
        object.__setattr__(self, "aliases", tuple(normalized_aliases))

    @property
    def canonical_model(self) -> str:
        return CANONICAL_NEMOTRON_MODEL

    def matches(self, advertised_model: str) -> bool:
        normalized = _normalize_identifier(advertised_model)
        if not normalized:
            return False
        accepted_exact = {
            _normalize_identifier(self.configured_model),
            *(_normalize_identifier(alias) for alias in self.aliases),
        }
        return normalized in accepted_exact or _is_supported_family_identifier(
            advertised_model
        )

    def select_advertised(self, advertised_models: list[str]) -> str | None:
        valid = [model for model in advertised_models if self.matches(model)]
        if not valid:
            return None
        preferred = [self.configured_model, *self.aliases]
        for candidate in preferred:
            normalized = _normalize_identifier(candidate)
            exact = next(
                (
                    model
                    for model in valid
                    if _normalize_identifier(model) == normalized
                ),
                None,
            )
            if exact is not None:
                return exact
        return valid[0]


def _is_supported_family_identifier(value: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(value).casefold()))
    return _REQUIRED_FAMILY_TOKENS.issubset(tokens)


def _normalize_identifier(value: str) -> str:
    return "/".join(re.findall(r"[a-z0-9]+", str(value).casefold()))
