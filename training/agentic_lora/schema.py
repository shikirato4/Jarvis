from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from typing import Any

SCHEMA_VERSION = "jarvis.agentic_lora.v1"
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
TOOL_STATUSES = {"success", "failed", "timeout", "confirmation_required", "blocked", "empty"}
PRIVATE_MARKERS = (
    "reasoning_content",
    "chain_of_thought",
    "raw_cot",
    "hidden_reasoning",
    "private_thoughts",
    "<think>",
    "</think>",
    "C:\\Users\\shiki",
    "C:\\\\Users\\\\shiki",
    ".env",
    "api_key",
    "access_token",
    "password=",
    "secret_key",
)
SUCCESS_WORDS = re.compile(r"\b(done|completed|finished|listo|hecho|completad[oa])\b", re.I)


class RecordError(ValueError):
    """A canonical record failed a deterministic quality gate."""


def wire_name(canonical_name: str) -> str:
    return canonical_name.replace(".", "__")


def canonical_name(wire: str) -> str:
    return wire.replace("__", ".")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def schema_fingerprint(tool: dict[str, Any]) -> str:
    """Hash a tool schema for split-leakage checks."""
    function = tool.get("function", {})
    payload = {"description": function.get("description", ""), "parameters": function.get("parameters", {})}
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _prompt_text(record: dict[str, Any]) -> str:
    return " ".join(
        message["content"].strip().lower()
        for message in record.get("messages", [])
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    )


def _shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[index : index + width]) for index in range(max(0, len(words) - width + 1))}


def find_near_duplicates(records: list[dict[str, Any]], threshold: float = 0.98) -> list[tuple[str, str, float]]:
    """Return likely accidental prompt duplicates; contrast pairs are exempt."""
    candidates: list[tuple[str, str, float]] = []
    indexed = [(record, _shingles(_prompt_text(record))) for record in records]
    for index, (left, left_shingles) in enumerate(indexed):
        for right, right_shingles in indexed[index + 1 :]:
            if not left_shingles or not right_shingles:
                continue
            if left.get("contrast_pair_id") and left.get("contrast_pair_id") == right.get("contrast_pair_id"):
                continue
            score = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
            if score >= threshold and left["record_id"] != right["record_id"]:
                candidates.append((left["record_id"], right["record_id"], round(score, 4)))
    return candidates


def estimate_length(record: dict[str, Any]) -> int:
    """Conservative character-to-token estimate without loading the tokenizer."""
    visible = [json.dumps(record.get("tools", []), ensure_ascii=False)]
    visible.extend(
        json.dumps(message, ensure_ascii=False)
        for message in record.get("messages", [])
        if message.get("role") != "system"
    )
    return max(1, sum(len(item) for item in visible) // 3)


def _schema_type_ok(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected = schema.get("type")
    if expected and not _schema_type_ok(value, expected):
        raise RecordError(f"{path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise RecordError(f"{path}: value is outside enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise RecordError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise RecordError(f"{path}: above maximum")
    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise RecordError(f"{path}: shorter than minLength")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - value.keys()
        if missing:
            raise RecordError(f"{path}: missing required {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - properties.keys()
            if extras:
                raise RecordError(f"{path}: forbidden keys {sorted(extras)}")
        for key, item in value.items():
            if key in properties:
                validate_arguments(item, properties[key], f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            validate_arguments(item, schema["items"], f"{path}[{index}]")


def tool_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tool in record.get("tools", []):
        function = tool.get("function", {})
        name = function.get("name")
        if isinstance(name, str):
            result[name] = function
    return result


def _scan_for_private_content(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False).lower()
    for marker in PRIVATE_MARKERS:
        if marker.lower() in text:
            raise RecordError(f"private or hidden reasoning marker: {marker}")


def validate_record(record: dict[str, Any], *, max_estimated_tokens: int = 1024) -> None:
    if not isinstance(record, dict):
        raise RecordError("record must be an object")
    required = {"schema_version", "record_id", "split", "locale", "category_tags", "tools", "messages", "expected_behavior"}
    missing = required - record.keys()
    if missing:
        raise RecordError(f"missing top-level fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise RecordError("unsupported schema_version")
    if not re.fullmatch(r"(train|validation|heldout|hard_regression)_[a-z0-9_]+", record["record_id"]):
        raise RecordError("record_id is not deterministic/canonical")
    if record["split"] not in {"train", "validation", "heldout", "hard_regression"}:
        raise RecordError("invalid split")
    if not isinstance(record["category_tags"], list) or not record["category_tags"]:
        raise RecordError("category_tags must be non-empty")
    if not isinstance(record["tools"], list):
        raise RecordError("tools must be a list")
    functions = tool_map(record)
    if len(functions) != len(record["tools"]):
        raise RecordError("tool names must be unique")
    for name, function in functions.items():
        if "." not in name or not function.get("description") or not isinstance(function.get("parameters"), dict):
            raise RecordError(f"invalid tool schema: {name}")
    messages = record["messages"]
    if not isinstance(messages, list) or not messages:
        raise RecordError("messages must be non-empty")
    call_ids: set[str] = set()
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    observations: dict[str, dict[str, Any]] = {}
    for message in messages:
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise RecordError(f"invalid role: {role}")
        if role == "assistant" and "tool_calls" in message:
            if message.get("content") not in (None, ""):
                raise RecordError("tool-call assistant message must not add prose")
            for call in message["tool_calls"]:
                function = call.get("function", {})
                call_id = call.get("id")
                name = function.get("name")
                if not call_id or call_id in call_ids or name not in functions:
                    raise RecordError("invalid or duplicate assistant tool call")
                arguments = function.get("arguments")
                if not isinstance(arguments, dict):
                    raise RecordError("source arguments must be an object")
                validate_arguments(arguments, functions[name].get("parameters", {}), f"tool_call.{name}")
                call_ids.add(call_id)
                calls[call_id] = (name, arguments)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in calls or message.get("name") != calls[call_id][0]:
                raise RecordError("tool observation does not link to preceding call")
            content = message.get("content")
            if not isinstance(content, dict) or content.get("action") != message["name"]:
                raise RecordError("invalid observation envelope")
            if content.get("status") not in TOOL_STATUSES:
                raise RecordError("invalid observation status")
            observations[call_id] = content
        elif role in {"system", "user", "assistant"} and not isinstance(message.get("content"), str):
            raise RecordError("text message content must be a string")
    if record.get("expected_behavior", {}).get("requires_tool") and not calls:
        raise RecordError("required-tool record has no tool call")
    for observation in observations.values():
        if observation["status"] in {"failed", "timeout", "confirmation_required", "blocked"}:
            final_text = " ".join(
                message.get("content", "")
                for message in messages
                if message.get("role") == "assistant" and isinstance(message.get("content"), str)
            )
            if SUCCESS_WORDS.search(final_text):
                raise RecordError("false-success claim after non-success observation")
    final_messages = [message for message in messages if message.get("role") == "assistant" and isinstance(message.get("content"), str)]
    if not final_messages:
        raise RecordError("record must have a non-empty assistant final")
    if any(not message["content"].strip() for message in final_messages):
        raise RecordError("assistant final must be non-empty")
    _scan_for_private_content(record)
    if estimate_length(record) > max_estimated_tokens:
        raise RecordError("record exceeds conservative pilot length limit")


def validate_dataset(records: list[dict[str, Any]], *, split_counts: dict[str, int] | None = None) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_prompts: dict[str, str] = {}
    for record in records:
        try:
            validate_record(record)
        except RecordError as exc:
            errors.append(f"{record.get('record_id', '<unknown>')}: {exc}")
            continue
        record_id = record["record_id"]
        if record_id in seen_ids:
            errors.append(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        prompt = _prompt_text(record)
        previous = seen_prompts.get(prompt)
        if previous and previous != record_id:
            errors.append(f"duplicate user prompt: {previous} and {record_id}")
        seen_prompts[prompt] = record_id
    if split_counts is not None:
        actual = Counter(record.get("split") for record in records)
        for split, expected in split_counts.items():
            if actual[split] != expected:
                errors.append(f"split {split}: expected {expected}, got {actual[split]}")
    heldout_families = {"spaceship", "observatory", "aquarium", "museum", "drone_sim", "energy_grid_sim"}
    heldout_fingerprints: set[str] = set()
    for record in records:
        if record.get("split") == "heldout":
            heldout_fingerprints.update(schema_fingerprint(tool) for tool in record.get("tools", []))
    for record in records:
        family = record.get("tool_family", "")
        if family in heldout_families and record.get("split") not in {"heldout"}:
            errors.append(f"held-out family leakage: {family} in {record.get('split')}")
        if record.get("split") in {"train", "validation"}:
            leaked = heldout_fingerprints.intersection(schema_fingerprint(tool) for tool in record.get("tools", []))
            if leaked:
                errors.append(f"held-out schema fingerprint leakage: {record['record_id']}")
    return errors
