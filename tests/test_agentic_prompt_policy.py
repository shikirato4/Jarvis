from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from jarvis.agent_runtime.prompt_policy import (
    CAPABILITY_UNIVERSE,
    CAPABILITY_MARKER,
    CORE_MARKER,
    FULL_CHARTER_MARKER,
    PERMANENT_CORE_CONTRACT,
    PRE_LORA_AGENTIC_SCAFFOLD,
    SCHEMA_MARKER,
    SCAFFOLD_MARKER,
    AgenticPromptPolicy,
    AgenticPromptProfile,
)
from jarvis.config import Settings
from jarvis.models_runtime.base import ModelMessage


def _tool(name: str, description: str = "Perform the described operation.") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    }


def test_prompt_composition_order_and_single_system_policy() -> None:
    policy = AgenticPromptPolicy(profile="compact")
    messages = [
        ModelMessage(role="system", content="Existing identity."),
        ModelMessage(role="user", content="Do it."),
    ]

    composed = policy.compose(messages, [_tool("novel.execute")])

    assert [message.role for message in composed] == ["system", "user"]
    system = composed[0].content
    assert system.index("Existing identity.") < system.index(CORE_MARKER)
    assert system.index(CORE_MARKER) < system.index(SCAFFOLD_MARKER)
    assert system.index(SCAFFOLD_MARKER) < system.index(CAPABILITY_MARKER)
    assert system.count(CORE_MARKER) == 1
    assert system.count(SCAFFOLD_MARKER) == 1


def test_pre_lora_scaffold_can_be_disabled_without_removing_core() -> None:
    composed = AgenticPromptPolicy(pre_lora_enabled=False).compose([], [])

    assert CORE_MARKER in composed[0].content
    assert SCAFFOLD_MARKER not in composed[0].content
    assert CAPABILITY_MARKER in composed[0].content


def test_pre_lora_feature_flag_is_configurable() -> None:
    settings = Settings(
        _env_file=None,
        pre_lora_agentic_scaffold_enabled=False,
    )

    assert settings.pre_lora_agentic_scaffold_enabled is False
    assert Settings(_env_file=None).pre_lora_agentic_profile == "full"


def test_recomposition_replaces_managed_layers_without_duplication() -> None:
    policy = AgenticPromptPolicy(profile="compact")
    first = policy.compose(
        [ModelMessage(role="system", content="Identity")],
        [_tool("alpha.read")],
    )
    second = policy.compose(first, [_tool("beta.write")])

    assert second[0].content.count(CORE_MARKER) == 1
    assert second[0].content.count(SCAFFOLD_MARKER) == 1
    assert second[0].content.count(CAPABILITY_MARKER) == 1
    assert "alpha.read" not in second[0].content
    assert "beta.write (required=target)" in second[0].content


def test_dynamic_capability_context_comes_from_supplied_schemas() -> None:
    text = AgenticPromptPolicy.capability_context(
        [_tool("unseen.rotate", "Rotate a previously unknown device.")]
    )

    assert "unseen: AVAILABLE_VIA_TOOL (1 operations)" in text
    assert "calculator" not in text


def test_family_manifest_uses_all_tools_while_schema_context_is_selected() -> None:
    policy = AgenticPromptPolicy(profile="full")
    composed = policy.compose(
        [ModelMessage(role="user", content="Find the file")],
        [_tool("files.search")],
        all_tools=[_tool("files.search"), _tool("terminal.run")],
    )

    system = composed[0].content
    assert "files: AVAILABLE_VIA_TOOL" in system
    assert "terminal: AVAILABLE_VIA_TOOL" in system
    assert "files.search (required=target)" in system
    selected = system.split(SCHEMA_MARKER, 1)[1]
    assert "terminal.run" not in selected


def test_large_registry_progressively_discloses_unseen_relevant_schema() -> None:
    tools = [_tool(f"noise.operation_{index}") for index in range(40)]
    tools.append(_tool("orbital.align", "Align an orbital antenna by target."))
    policy = AgenticPromptPolicy(max_relevant_tools=12)

    selected = policy.select_relevant_tools(
        tools, "Please align the orbital antenna to Vega."
    )

    names = [schema["function"]["name"] for schema in selected]
    assert "orbital.align" in names
    assert len(selected) <= 12


def test_large_registry_matches_related_schema_word_forms() -> None:
    tools = [_tool(f"noise.operation_{index}") for index in range(40)]
    tools.append(_tool("calculator.evaluate", "Evaluate arithmetic safely."))

    selected = AgenticPromptPolicy(max_relevant_tools=12).select_relevant_tools(
        tools, "Calculate 2 + 2."
    )

    assert any(
        schema["function"]["name"] == "calculator.evaluate" for schema in selected
    )


def test_static_contract_has_required_safety_and_truthfulness_invariants() -> None:
    static = f"{PERMANENT_CORE_CONTRACT}\n{PRE_LORA_AGENTIC_SCAFFOLD}".casefold()

    assert "securitygate" in static
    assert "ability is not permission" in static
    assert "never claim an action occurred" in static
    assert "private chain-of-thought" in static
    assert "untrusted data" in static
    assert "models are replaceable providers" in static
    assert not any(name in static for name in ("qwen", "nemotron", "kimi"))


def test_full_charter_represents_exact_200_conceptual_capabilities() -> None:
    group_lines = [
        line
        for line in CAPABILITY_UNIVERSE.splitlines()
        if line.startswith(("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X."))
    ]
    names = [
        name
        for line in group_lines
        for name in line.split(": ", 1)[1].rstrip(".").split("; ")
    ]

    assert len(names) == 200
    assert "Intent Reconstruction" in names
    assert "Controlled Self-Improvement Loop" in names
    assert "not tools active now" in CAPABILITY_UNIVERSE


def test_prompt_profiles_fit_production_token_budgets() -> None:
    metadata = Path("training/agentic_lora/base_metadata")
    tokenizer = AutoTokenizer.from_pretrained(metadata, local_files_only=True)
    counts = {
        profile: len(
            tokenizer.encode(AgenticPromptPolicy(profile=profile).static_prompt)
        )
        for profile in AgenticPromptProfile
    }

    assert counts[AgenticPromptProfile.OFF] < counts[AgenticPromptProfile.COMPACT]
    assert counts[AgenticPromptProfile.COMPACT] <= 1_600
    assert 2_000 <= counts[AgenticPromptProfile.FULL] <= 3_200
    assert FULL_CHARTER_MARKER in AgenticPromptPolicy().static_prompt
