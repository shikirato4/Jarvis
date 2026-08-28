"""Jarvis cognitive operating system core with a lightweight package import."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bootstrap import JarvisApplication

__all__ = ["JarvisApplication", "build_application"]


def __getattr__(name: str) -> Any:
    if name in {"JarvisApplication", "build_application"}:
        from . import bootstrap

        return getattr(bootstrap, name)
    raise AttributeError(f"module 'jarvis' has no attribute {name!r}")
