from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import Field

from jarvis.models.base import JarvisBaseModel

from .errors import MetaCommandParseError


class MetaCommandKind(StrEnum):
    HELP = "help"
    MODE = "mode"
    ACTION = "action"
    TOOL = "tool"
    TASK = "task"
    STATE = "state"
    REMEMBER = "remember"
    RECALL = "recall"


class MetaCommand(JarvisBaseModel):
    raw: str
    kind: MetaCommandKind
    target: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    arguments: list[str] = Field(default_factory=list)


class MetaCommandParser:
    def parse(self, raw_text: str) -> MetaCommand:
        text = raw_text.strip()
        if not text.startswith("/"):
            raise MetaCommandParseError("metacommands must start with '/'")

        command_name, _, remainder = text[1:].partition(" ")
        try:
            kind = MetaCommandKind(command_name.strip().lower())
        except ValueError as exc:
            raise MetaCommandParseError(f"unknown metacommand '{command_name}'") from exc

        if kind in {MetaCommandKind.HELP, MetaCommandKind.STATE}:
            return MetaCommand(raw=text, kind=kind)
        if kind == MetaCommandKind.MODE:
            return self._parse_mode(text, remainder)
        if kind == MetaCommandKind.ACTION:
            return self._parse_targeted(text, kind, remainder)
        if kind == MetaCommandKind.TOOL:
            return self._parse_targeted(text, kind, remainder)
        if kind == MetaCommandKind.TASK:
            return self._parse_targeted(text, kind, remainder)
        if kind == MetaCommandKind.REMEMBER:
            payload = {"content": remainder.strip(), "kind": "note", "source": "metacommand"}
            return MetaCommand(raw=text, kind=kind, payload=payload)
        if kind == MetaCommandKind.RECALL:
            payload = {"query": remainder.strip(), "limit": 10}
            return MetaCommand(raw=text, kind=kind, payload=payload)
        raise MetaCommandParseError(f"unsupported metacommand '{kind.value}'")

    def _parse_mode(self, raw: str, remainder: str) -> MetaCommand:
        if not remainder.strip():
            raise MetaCommandParseError("/mode requires a target mode")
        parts = remainder.split()
        target = parts[0]
        arguments = parts[1:]
        sticky = "--transient" not in arguments
        reason = " ".join(argument for argument in arguments if not argument.startswith("--")) or None
        return MetaCommand(
            raw=raw,
            kind=MetaCommandKind.MODE,
            target=target,
            payload={"sticky": sticky, "reason": reason},
            arguments=arguments,
        )

    def _parse_targeted(self, raw: str, kind: MetaCommandKind, remainder: str) -> MetaCommand:
        target, payload = self._split_target_payload(remainder)
        return MetaCommand(raw=raw, kind=kind, target=target, payload=payload)

    @staticmethod
    def _split_target_payload(remainder: str) -> tuple[str, dict[str, Any]]:
        target, _, payload_raw = remainder.strip().partition(" ")
        if not target:
            raise MetaCommandParseError("metacommand target is required")
        payload_text = payload_raw.strip()
        if not payload_text:
            return target, {}
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise MetaCommandParseError(f"invalid JSON payload: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise MetaCommandParseError("metacommand payload must be a JSON object")
        return target, parsed
