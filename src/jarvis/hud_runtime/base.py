from __future__ import annotations

from typing import Protocol


class HudSectionBuilder(Protocol):
    def build(self) -> dict[str, object]: ...


class HudRenderer(Protocol):
    def render_shell(self, *, title: str, poll_interval_ms: int) -> str: ...
