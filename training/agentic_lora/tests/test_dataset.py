from __future__ import annotations

from training.agentic_lora.generator import SPLIT_COUNTS, build_records
from training.agentic_lora.schema import RecordError, canonical_name, find_near_duplicates, validate_dataset, validate_record, wire_name


def test_pilot_counts_and_validation() -> None:
    records = build_records(20260831)
    assert len(records) == 1200
    assert validate_dataset(records, split_counts=SPLIT_COUNTS) == []


def test_deterministic_ids_and_content() -> None:
    first = build_records(20260831)
    second = build_records(20260831)
    assert first == second
    assert [item["record_id"] for item in first[:3]] == ["train_000000", "train_000001", "train_000002"]


def test_wire_name_round_trip() -> None:
    assert canonical_name(wire_name("garden.set_level")) == "garden.set_level"


def test_required_argument_rejected() -> None:
    record = next(item for item in build_records(20260831) if item["real_or_fictitious"] == "fictitious" and item["expected_behavior"]["requires_tool"])
    call = next(message for message in record["messages"] if message.get("tool_calls"))
    call["tool_calls"][0]["function"]["arguments"] = {}
    try:
        validate_record(record)
    except RecordError as exc:
        assert "missing required" in str(exc)
    else:
        raise AssertionError("invalid required arguments were accepted")


def test_heldout_families_are_not_train_or_validation() -> None:
    records = build_records(20260831)
    heldout = {"spaceship", "observatory", "aquarium", "museum", "drone_sim", "energy_grid_sim"}
    assert all(item["tool_family"] not in heldout for item in records if item["split"] in {"train", "validation"})


def test_private_marker_rejected() -> None:
    record = build_records(20260831)[0]
    record["messages"][1]["content"] = "C:\\Users\\shiki\\private.txt"
    try:
        validate_record(record)
    except RecordError as exc:
        assert "private" in str(exc)
    else:
        raise AssertionError("private path was accepted")


def test_near_duplicate_detector_and_contrast_exemption() -> None:
    records = build_records(20260831)
    left = records[0]
    right = records[1]
    right["messages"][1]["content"] = left["messages"][1]["content"].replace("caso 0", "caso 1")
    assert find_near_duplicates([left, right], threshold=0.8)
    right["contrast_pair_id"] = left["contrast_pair_id"] = "intentional-contrast"
    assert find_near_duplicates([left, right], threshold=0.8) == []


def test_exact_compiler_masks_prompt_and_observation() -> None:
    from transformers import AutoTokenizer
    from training.agentic_lora.compile_dataset import compile_record

    record = next(item for item in build_records(20260831) if item["expected_behavior"]["requires_tool"])
    tokenizer = AutoTokenizer.from_pretrained("training/agentic_lora/base_metadata", local_files_only=True, use_fast=True)
    compiled = compile_record(record, tokenizer)
    assert compiled["visible_token_count"] == len(compiled["input_ids"]) == len(compiled["labels"])
    assert compiled["assistant_label_token_count"] > 0
    assert compiled["assistant_label_token_count"] < compiled["visible_token_count"]
    assert -100 in compiled["labels"]


def test_golden_wire_serialization_keeps_observation_masked() -> None:
    from transformers import AutoTokenizer
    from training.agentic_lora.compile_dataset import _wire_messages, _wire_tools, compile_record

    record = next(item for item in build_records(20260831) if item["expected_behavior"]["requires_tool"])
    messages = _wire_messages(record["messages"])
    tools = _wire_tools(record["tools"])
    assert "." not in tools[0]["function"]["name"]
    assert "__" in tools[0]["function"]["name"]
    observation = next(message for message in messages if message["role"] == "tool")
    assert isinstance(observation["content"], str)
    assert '"status"' in observation["content"]
    tokenizer = AutoTokenizer.from_pretrained("training/agentic_lora/base_metadata", local_files_only=True, use_fast=True)
    compiled = compile_record(record, tokenizer)
    assert compiled["tool_schema_token_count"] > 0
    assert compiled["assistant_label_token_count"] < compiled["visible_token_count"]
