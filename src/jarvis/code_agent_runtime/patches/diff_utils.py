from __future__ import annotations

import difflib
import hashlib


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unified_diff(path: str, before: str, after: str, *, max_chars: int = 20_000) -> tuple[str, bool]:
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    text = "\n".join(lines)
    truncated = len(text) > max_chars
    return (text[:max_chars] + "\n... [diff truncated]" if truncated else text, truncated)
