from __future__ import annotations

from jarvis.identity import BASE_IDENTITY_PROMPT, jarvis_identity_prompt


def test_identity_addresses_the_user_as_senor() -> None:
    """JARVIS speaks with a courteous butler persona by default."""
    lowered = BASE_IDENTITY_PROMPT.casefold()

    assert "señor" in lowered
    # The instruction is behavioral guidance, not a literal echo requirement.
    assert "refiérete" in lowered or "refierete" in lowered


def test_identity_extra_context_appends_after_base() -> None:
    prompt = jarvis_identity_prompt("Superficie: CLI local.")

    assert "señor" in prompt.casefold()
    assert prompt.endswith("Superficie: CLI local.")
    assert BASE_IDENTITY_PROMPT in prompt
