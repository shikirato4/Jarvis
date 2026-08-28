from __future__ import annotations

import json
from typing import Protocol

from jarvis.models_runtime.base import ModelMessage


class TokenEstimator(Protocol):
    def estimate_text(self, text: str) -> int: ...

    def estimate_message(self, message: ModelMessage) -> int: ...


class ConservativeTokenEstimator:
    """Tokenizer-independent upper estimate suitable for context selection."""

    _MESSAGE_OVERHEAD_TOKENS = 4

    def estimate_text(self, text: str) -> int:
        return max((len(text.encode("utf-8")) + 3) // 4, 1)

    def estimate_message(self, message: ModelMessage) -> int:
        total = self._MESSAGE_OVERHEAD_TOKENS + self.estimate_text(message.content)
        if message.name:
            total += self.estimate_text(message.name)
        if message.tool_call_id:
            total += self.estimate_text(message.tool_call_id)
        for call in message.tool_calls:
            total += self.estimate_text(call.id)
            total += self.estimate_text(call.name)
            arguments = (
                call.arguments
                if isinstance(call.arguments, str)
                else json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            total += self.estimate_text(arguments)
        return total
