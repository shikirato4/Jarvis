"""Identity matching for explicitly configured local model endpoints."""
from __future__ import annotations

import re
from dataclasses import dataclass


def _normalize(value: str) -> str:
    return "/".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


@dataclass(frozen=True, slots=True)
class ConfiguredModelIdentity:
    """Match the configured model ID and aliases without assuming a family."""

    configured_model: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        configured = str(self.configured_model).strip()
        if (
            not configured
            or len(configured) > 512
            or any(ord(character) < 32 for character in configured)
        ):
            raise ValueError("configured local model identifier is invalid")
        aliases: list[str] = []
        seen = {_normalize(configured)}
        for raw in self.aliases:
            alias = str(raw).strip()
            key = _normalize(alias)
            if alias and key not in seen:
                seen.add(key)
                aliases.append(alias)
        object.__setattr__(self, "configured_model", configured)
        object.__setattr__(self, "aliases", tuple(aliases))

    @property
    def canonical_model(self) -> str:
        return self.configured_model

    def is_compatible(self, advertised_model: str) -> bool:
        key = _normalize(advertised_model)
        return key in {
            _normalize(self.configured_model),
            *(_normalize(alias) for alias in self.aliases),
        }

    def select_advertised(self, advertised_models: list[str]) -> str | None:
        for preferred in (self.configured_model, *self.aliases):
            key = _normalize(preferred)
            for model in advertised_models:
                if _normalize(model) == key:
                    return model
        return None
