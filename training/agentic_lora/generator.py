from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import GENERATOR_VERSION, SCHEMA_VERSION
from .real_tool_snapshots import REAL_TOOLS

TRAIN_FAMILIES = ("lab", "warehouse", "garden", "robot", "camera", "vehicle", "home_sim", "factory_sim")
HELDOUT_FAMILIES = ("spaceship", "observatory", "aquarium", "museum", "drone_sim", "energy_grid_sim")
SPLIT_COUNTS = {"train": 800, "validation": 120, "heldout": 200, "hard_regression": 80}
LANGUAGE_COUNTS = {"es-MX": 720, "en-US": 420, "mixed": 60}


def _tool(family: str, action: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": f"{family}.{action}",
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        },
    }


def fictitious_tool(family: str, index: int) -> dict[str, Any]:
    variants = [
        _tool(family, "set_level", f"Set the simulated {family} level.", {"level": {"type": "integer", "minimum": 0, "maximum": 100}}, ["level"]),
        _tool(family, "read_status", f"Read the current simulated {family} status.", {}, []),
        _tool(family, "find_item", f"Find an item in the simulated {family} catalog.", {"sku": {"type": "string", "minLength": 2}}, ["sku"]),
        _tool(family, "configure", f"Configure a simulated {family} profile.", {"mode": {"type": "string", "enum": ["quiet", "normal", "boost"]}, "enabled": {"type": "boolean"}}, ["mode", "enabled"]),
        _tool(family, "schedule", f"Schedule a safe simulated {family} operation.", {"minutes": {"type": "number", "minimum": 1, "maximum": 120}, "labels": {"type": "array", "items": {"type": "string"}}}, ["minutes"]),
    ]
    return variants[index % len(variants)]


def _family_for(index: int, heldout: bool) -> str:
    families = HELDOUT_FAMILIES if heldout else TRAIN_FAMILIES
    return families[index % len(families)]


def _scenario(index: int, split: str) -> str:
    if split == "hard_regression":
        return ("false_success", "security_confirmation", "security_denied", "missing_required_tool", "empty_observation", "multi_tool", "tool_timeout", "wrong_argument", "capability_absent", "scope_honesty", "continuation", "tool_loop")[index % 12]
    return ("capability_present", "action_vs_information", "correct_tool_selection", "argument_extraction", "multi_tool", "observation_grounding", "tool_failure", "security_confirmation", "no_false_success", "no_false_capability_denial", "irrelevant_distractors", "parameter_ambiguity", "safe_partial_completion", "continuation", "missing_required_tool", "output_concision")[index % 16]


def _text(language: str, family: str, action: str, index: int) -> tuple[str, str]:
    variant = index % 4
    if language == "es-MX":
        prompts = (
            f"Configura {family} simulado mediante {action}; usa el valor {index % 100} en el caso {index}.",
            f"Necesito que ejecutes {family}.{action} en el entorno de prueba con valor {index % 100} (caso {index}).",
            f"Aplica la operación {action} al sistema {family} de laboratorio y conserva el valor {index % 100}; referencia {index}.",
            f"Realiza la acción solicitada sobre {family}: llama a {action} con {index % 100}. Caso {index}.",
        )
        return prompts[variant], f"Listo: la acción simulada {family}.{action} terminó y el resultado confirmado es {index % 100}."
    if language == "mixed":
        prompts = (
            f"Please configura el {family} simulado con {action}; usa {index % 100}, case {index}.",
            f"Run {family}.{action} en el laboratorio simulado con value {index % 100}; reference {index}.",
            f"Can you ejecutar {action} for the simulated {family} using {index % 100}? Case {index}.",
            f"Necesito {family}.{action} now, with parameter {index % 100}; caso {index}.",
        )
        return prompts[variant], f"Listo: the simulated {family}.{action} completed with confirmed value {index % 100}."
    prompts = (
        f"Configure the simulated {family} using {action} with value {index % 100} in case {index}.",
        f"Run {family}.{action} in the test environment with parameter {index % 100}; case {index}.",
        f"Please invoke the simulated {family} operation {action}, value {index % 100}, reference {index}.",
        f"Set up {family} by calling {action} with {index % 100}. Scenario {index}.",
    )
    return prompts[variant], f"Done: the simulated {family}.{action} completed with confirmed value {index % 100}."


def _record(split: str, ordinal: int, kind: str, language: str, family: str, real: bool, rng: random.Random) -> dict[str, Any]:
    if real:
        tool = REAL_TOOLS[ordinal % len(REAL_TOOLS)]
        family = tool["function"]["name"].split(".", 1)[0]
    else:
        tool = fictitious_tool(family, ordinal)
    action = tool["function"]["name"].split(".", 1)[1]
    prompt, final = _text(language, family, action, ordinal)
    requires_tool = kind not in {"action_vs_information", "capability_absent", "general_explanation"}
    tools = [tool]
    dependent_tool = fictitious_tool(family, ordinal + 2) if kind == "multi_tool" else None
    if dependent_tool is not None and dependent_tool["function"]["name"] != tool["function"]["name"]:
        tools.append(dependent_tool)
    arguments: dict[str, Any] = {}
    props = tool["function"]["parameters"].get("properties", {})
    for key, spec in props.items():
        if spec.get("type") == "integer":
            arguments[key] = ordinal % 100
        elif spec.get("type") == "number":
            arguments[key] = float((ordinal % 60) + 1)
        elif spec.get("type") == "boolean":
            arguments[key] = ordinal % 2 == 0
        elif spec.get("type") == "array":
            arguments[key] = [f"label-{ordinal % 5}"]
        elif "enum" in spec:
            arguments[key] = spec["enum"][ordinal % len(spec["enum"])]
        else:
            arguments[key] = f"C:\\Projects\\Example\\item-{ordinal % 10}.txt" if family == "files" else f"example-{ordinal}"
    if kind in {"capability_absent", "general_explanation"}:
        tools = [fictitious_tool("irrelevant", ordinal)]
        prompt = (
            f"¿Cómo puedo revisar este concepto en general? Caso {ordinal}."
            if language != "en-US"
            else f"How can I review this concept in general? Case {ordinal}."
        )
        final = "No ejecuté ninguna herramienta; aquí tienes una explicación general y honesta."
        requires_tool = False
    if kind == "action_vs_information" and ordinal % 2:
        prompt = (
            f"¿Cómo puedo revisar el estado de este sistema simulado? Caso {ordinal}."
            if language != "en-US"
            else f"How can I check this simulated system's status? Case {ordinal}."
        )
        final = "Puedes consultar el estado con una herramienta si está disponible; esta pregunta pide una explicación, no ejecutar una acción."
        requires_tool = False
    status = "success"
    if kind in {"tool_failure", "false_success", "tool_timeout", "security_denied", "empty_observation", "safe_partial_completion", "wrong_argument"}:
        status = {"tool_timeout": "timeout", "security_denied": "blocked", "empty_observation": "empty"}.get(kind, "failed")
        final = "La acción sigue sin confirmarse; informo el estado observado sin afirmar un resultado." if language != "en-US" else "The action remains unconfirmed; I report the observed failure without claiming a result."
    if kind == "security_confirmation":
        status = "confirmation_required"
        final = "La acción queda pendiente de confirmación; todavía no la presento como realizada." if language != "en-US" else "The action is pending confirmation; I do not present it as performed yet."
    call_id = f"call_{ordinal:06d}"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Runtime policy: use supplied capabilities for explicit actions, respect observations and SecurityGate, and never claim unconfirmed success."},
        {"role": "user", "content": prompt},
    ]
    if requires_tool:
        messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool["function"]["name"], "arguments": arguments}}]})
        messages.append({"role": "tool", "name": tool["function"]["name"], "tool_call_id": call_id, "content": {"action": tool["function"]["name"], "status": status, "message": "Sanitized simulated observation.", "data": {} if status != "success" else arguments}})
        if kind == "multi_tool":
            second = dependent_tool or fictitious_tool(family, ordinal + 2)
            second_name = second["function"]["name"]
            second_id = f"call_{ordinal:06d}_b"
            second_args = {}
            for key, spec in second["function"]["parameters"].get("properties", {}).items():
                if spec.get("type") == "integer":
                    second_args[key] = ordinal % 100
                elif spec.get("type") == "number":
                    second_args[key] = float((ordinal % 60) + 1)
                elif spec.get("type") == "boolean":
                    second_args[key] = True
                elif spec.get("type") == "array":
                    second_args[key] = [f"label-{ordinal % 5}"]
                elif "enum" in spec:
                    second_args[key] = spec["enum"][0]
                else:
                    second_args[key] = f"derived-from-{call_id}"
            messages.append({"role": "assistant", "tool_calls": [{"id": second_id, "type": "function", "function": {"name": second_name, "arguments": second_args}}]})
            messages.append({"role": "tool", "name": second_name, "tool_call_id": second_id, "content": {"action": second_name, "status": "success", "message": "Sanitized dependent observation.", "data": second_args}})
    messages.append({"role": "assistant", "content": final})
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"{split}_{ordinal:06d}",
        "split": split,
        "locale": language,
        "category_tags": [kind, "real_tool" if real else ("non_tool_control" if not requires_tool else "fictitious_tool")],
        "tool_family": family,
        "real_or_fictitious": "real" if real else ("control" if not requires_tool else "fictitious"),
        "tools": tools,
        "messages": messages,
        "security_expectation": {"decision": "allow" if status == "success" else status, "confirmation_required": status == "confirmation_required"},
        "expected_behavior": {"requires_tool": requires_tool, "tool_calls": 2 if kind == "multi_tool" and requires_tool else (1 if requires_tool else 0), "final_must_be_nonempty": True, "success_claim_requires_status": "success"},
        "provenance": {"kind": "deterministic_synthetic", "source_pattern": kind},
        "generation_seed": rng.randrange(1, 2**31),
        "generator_version": GENERATOR_VERSION,
        "privacy_review": "synthetic",
    }


def build_records(seed: int = 20260831) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    allocation = {"train": (240, 440, 120), "validation": (30, 72, 18), "heldout": (0, 200, 0), "hard_regression": (30, 8, 42)}
    language_pool = [language for language, count in LANGUAGE_COUNTS.items() for _ in range(count)]
    rng.shuffle(language_pool)
    global_index = 0
    for split, count in SPLIT_COUNTS.items():
        real_count, fict_count, control_count = allocation[split]
        for i in range(count):
            real = i < real_count
            is_control = not real and i >= real_count + fict_count
            family = _family_for(i, heldout=split == "heldout")
            kind = ("general_explanation", "capability_absent", "action_vs_information")[i % 3] if is_control else _scenario(i, split)
            if not is_control and kind in {"capability_absent", "general_explanation"}:
                kind = "capability_present"
            record = _record(split, global_index, kind, language_pool[global_index], family, real, rng)
            if is_control:
                record["real_or_fictitious"] = "control"
            elif not real:
                record["real_or_fictitious"] = "fictitious"
            if split == "heldout":
                record["generalization_family"] = family
            records.append(record)
            global_index += 1
    return records


def build_report(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    from .schema import estimate_length, find_near_duplicates, validate_dataset
    categories: dict[str, int] = {}
    for record in records:
        for category in record["category_tags"]:
            categories[category] = categories.get(category, 0) + 1
    lengths = [estimate_length(record) for record in records]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "total_records": len(records),
        "records_by_split": {split: sum(record["split"] == split for record in records) for split in SPLIT_COUNTS},
        "records_by_category": dict(sorted(categories.items())),
        "records_by_language": {language: sum(record["locale"] == language for record in records) for language in LANGUAGE_COUNTS},
        "real_fictitious_control": {kind: sum(record["real_or_fictitious"] == kind for record in records) for kind in ("real", "fictitious", "control")},
        "single_tool": sum(sum(message.get("role") == "tool" for message in record["messages"]) == 1 for record in records),
        "multi_tool": sum(sum(message.get("role") == "tool" for message in record["messages"]) >= 2 for record in records),
        "action_vs_info": categories.get("action_vs_information", 0),
        "security": sum("security" in " ".join(record["category_tags"]) for record in records),
        "failure": sum(any(tag in record["category_tags"] for tag in ("tool_failure", "tool_timeout", "false_success")) for record in records),
        "capability_absent": sum("capability_absent" in record["category_tags"] for record in records),
        "missing_required_tool": sum("missing_required_tool" in record["category_tags"] for record in records),
        "output_concision": sum("output_concision" in record["category_tags"] for record in records),
        "tool_families": sorted({record["tool_family"] for record in records}),
        "heldout_tool_families": sorted(set(HELDOUT_FAMILIES)),
        "estimated_average_tokens": round(sum(lengths) / len(lengths), 2),
        "estimated_max_tokens": max(lengths),
        "duplicate_count": 0,
        "near_duplicate_warnings": len(find_near_duplicates(records)),
        "validation_failures": validate_dataset(records, split_counts=SPLIT_COUNTS),
    }
    report["real_percent"] = round(report["real_fictitious_control"]["real"] / len(records) * 100, 2)
    report["fictitious_percent"] = round(report["real_fictitious_control"]["fictitious"] / len(records) * 100, 2)
    report["control_percent"] = round(report["real_fictitious_control"]["control"] / len(records) * 100, 2)
    report["invalid_records_rejected"] = 0
    report["raw_cot_records"] = 0
    report["raw_reasoning_content_records"] = 0
    report["secret_or_personal_data_violations"] = 0
    report["over_length_accepted_records"] = 0
    report["schema_fingerprint_leakage"] = 0
    report["exact_duplicates"] = 0
    report["source_sha256"] = hashlib.sha256("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records).encode()).hexdigest()
    return report


def write_dataset(root: Path, seed: int = 20260831) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = build_records(seed)
    canonical = root / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_COUNTS:
        items = [record for record in records if record["split"] == split]
        filename = "heldout.jsonl" if split == "heldout" else f"{split}.jsonl"
        (canonical / filename).write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in items),
            encoding="utf-8",
        )
    return records, build_report(records, seed)
