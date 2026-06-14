from __future__ import annotations

import re
from dataclasses import dataclass


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b(?:api[_\s-]?key|token|password|passwd|pwd|credential|secret|pin)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:OPENAI|GEMINI|GOOGLE|BRAVE)[_A-Z0-9]*API[_-]?KEY\b", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:\.env|\.env\.[\w-]+)(?:\s|$)", re.IGNORECASE),
    re.compile(r"\b(?:id_rsa|\.pem|\.key|certificate|private key)\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:[\\/](?:Users|Windows|Program Files|ProgramData)[\\/][^\s]+", re.IGNORECASE),
    re.compile(r"(?:^|\s)(?:\.\.[\\/])+", re.IGNORECASE),
)

_PRIVATE_CODE_MARKERS = (
    "def ",
    "class ",
    "import ",
    "from ",
    "function ",
    "const ",
    "let ",
    "var ",
    "{",
    "}",
)


@dataclass(frozen=True)
class SanitizedQuery:
    allowed: bool
    query: str
    reason: str = ""


def sanitize_web_query(query: str, *, max_chars: int = 240) -> SanitizedQuery:
    compact = " ".join((query or "").strip().split())
    if not compact:
        return SanitizedQuery(False, "", "empty query")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(compact):
            return SanitizedQuery(False, "", "query may contain secrets or local private paths")
    if len(compact) > 800 and any(marker in compact for marker in _PRIVATE_CODE_MARKERS):
        return SanitizedQuery(False, "", "query looks like private file content")
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit(" ", 1)[0] or compact[:max_chars]
    return SanitizedQuery(True, compact)
