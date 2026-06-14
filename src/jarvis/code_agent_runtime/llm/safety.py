from __future__ import annotations

from jarvis.code_agent_runtime.llm.response_parser import ALLOWED_OPERATIONS

SECURITY_RULES = (
    "LLM providers never receive direct tool access.",
    "LLM output must be parsed into explicit patch operations.",
    "Secrets, sensitive files and paths outside the project are blocked.",
    "Patch application remains a separate permission-gated flow.",
)

__all__ = ["ALLOWED_OPERATIONS", "SECURITY_RULES"]
