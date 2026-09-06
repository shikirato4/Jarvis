from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.models_runtime.local_profiles import (
    BASELINE_PROFILE_ID,
    QWEN35_PROFILE_ID,
    LocalModelProfile,
    LocalModelProfileConfigurationError,
    baseline_local_model_profile,
    resolve_local_model_profile,
    registered_local_model_profiles,
)
from jarvis.models_runtime.qwen_identity import InvalidQwenIdentityError, QwenIdentity
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.qwen_local import QwenLocalProvider


def test_profile_is_immutable_and_validates_core_fields(tmp_path: Path) -> None:
    profile = LocalModelProfile(
        id="candidate",
        display_name="Candidate",
        model_alias="Qwen3.5-9B-Q4",
        model_aliases=(),
        gguf_path=tmp_path / "candidate.gguf",
        expected_model_family="qwen3.5",
        context_size=8192,
        parallel=1,
    )

    with pytest.raises((AttributeError, TypeError)):
        profile.context_size = 16_384  # type: ignore[misc]


def test_default_and_explicit_resolution_promote_qwen35(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=BASELINE_PROFILE_ID,
    )

    default = resolve_local_model_profile(Settings(_env_file=None, workspace_root=tmp_path))
    explicit = resolve_local_model_profile(settings)

    assert default.id == QWEN35_PROFILE_ID
    assert explicit.id == BASELINE_PROFILE_ID
    assert default.model_alias == "Qwen3.5-9B-IQ4_XS"
    assert default.context_size == 8192
    assert default.parallel == 1


def test_baseline_is_synthesized_from_current_settings(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_llm_model="Qwen3.8-custom",
        local_llm_model_aliases=("Qwen3.8-custom-alias",),
        local_runtime_model_path=Path("models/custom/model.gguf"),
        local_runtime_context_size=12_345,
        local_runtime_parallel=2,
    )

    profile = baseline_local_model_profile(settings)

    assert profile.id == BASELINE_PROFILE_ID
    assert profile.model_alias == "Qwen3.8-custom"
    assert profile.model_aliases == ("Qwen3.8-custom-alias",)
    assert profile.gguf_path == Path("models/custom/model.gguf")
    assert profile.context_size == 12_345
    assert profile.parallel == 2


def test_unknown_profile_fails_closed(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id="missing-profile",
    )

    with pytest.raises(LocalModelProfileConfigurationError, match="missing-profile"):
        resolve_local_model_profile(settings)


def test_qwen35_candidate_is_registered_as_default(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, workspace_root=tmp_path)
    candidate = next(
        profile
        for profile in registered_local_model_profiles(settings)
        if profile.id == QWEN35_PROFILE_ID
    )

    assert candidate.gguf_path.as_posix() == (
        "models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf"
    )
    assert candidate.model_alias == "Qwen3.5-9B-IQ4_XS"
    assert candidate.expected_model_family == "qwen3.5"
    assert candidate.context_size == 8192
    assert candidate.parallel == 1
    assert candidate.vision_capable is False
    assert candidate.audio_capable is False
    assert candidate.video_capable is False
    assert candidate.native_transport_modalities == ()
    assert candidate.reasoning_adapter == "qwen_enable_thinking_binary"
    assert resolve_local_model_profile(settings).id == QWEN35_PROFILE_ID
    selected = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=QWEN35_PROFILE_ID,
    )
    assert resolve_local_model_profile(selected).id == QWEN35_PROFILE_ID


def test_qwen38_rollback_remains_explicitly_selectable(tmp_path: Path) -> None:
    selected = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=BASELINE_PROFILE_ID,
    )
    profile = resolve_local_model_profile(selected)
    assert profile.id == BASELINE_PROFILE_ID
    assert profile.model_alias == "Qwen3.8-27B-IQ4_XS"
    assert profile.context_size == 4096


def test_qwen35_context_override_is_isolated_from_baseline(tmp_path: Path) -> None:
    candidate_settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=QWEN35_PROFILE_ID,
        local_runtime_context_override=16_384,
    )
    candidate = resolve_local_model_profile(candidate_settings)
    baseline = resolve_local_model_profile(
        Settings(
            _env_file=None,
            workspace_root=tmp_path,
            local_runtime_profile_id=BASELINE_PROFILE_ID,
            local_runtime_context_override=16_384,
        )
    )

    assert candidate.context_size == 16_384
    assert baseline.id == BASELINE_PROFILE_ID
    assert baseline.context_size == 4096


def test_explicit_registered_profile_resolves_without_becoming_default(tmp_path: Path) -> None:
    candidate = LocalModelProfile(
        id="future-qwen-profile",
        display_name="Future Qwen profile",
        model_alias="Qwen3.5-future",
        gguf_path=tmp_path / "future.gguf",
        expected_model_family="qwen3.5",
        context_size=16_384,
        parallel=1,
    )
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=candidate.id,
    )

    assert resolve_local_model_profile(settings, (candidate,)) is candidate
    assert resolve_local_model_profile(
        Settings(_env_file=None, workspace_root=tmp_path), (candidate,)
    ).id == QWEN35_PROFILE_ID


@pytest.mark.parametrize(
    ("value", "family"),
    [("Qwen3.8-27B-IQ4_XS", "qwen3.8"), ("Qwen3.5-9B-Q4", "qwen3.5")],
)
def test_qwen_identity_distinguishes_model_families(value: str, family: str) -> None:
    identity = QwenIdentity(value, expected_model_family=family)
    assert identity.is_compatible(value)


def test_qwen_identity_does_not_infer_qwen_38_from_8b_size_label() -> None:
    with pytest.raises(InvalidQwenIdentityError):
        QwenIdentity("Qwen3-8B-GGUF", expected_model_family="qwen3.8")


def test_reserved_llama_cpp_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        LocalModelProfile(
            id="bad",
            display_name="Bad",
            model_alias="Qwen3.5-9B",
            gguf_path=tmp_path / "bad.gguf",
            expected_model_family="qwen3.5",
            context_size=8192,
            parallel=1,
            llama_cpp_args=(("--ctx-size", "999999"),),
        )


@pytest.mark.parametrize(
    "prompt",
    [
        "Revisa qué versión de Python tengo.",
        "Comprueba mi versión de Python.",
        "Ejecuta Python y dime qué versión está instalada.",
    ],
)
def test_qwen35_terminal_action_retains_terminal_tools(tmp_path: Path, prompt: str) -> None:
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=QWEN35_PROFILE_ID,
    )
    provider = QwenLocalProvider(settings)
    request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content=prompt)],
        tools=[
            {"type": "function", "function": {"name": "terminal.run"}},
            {"type": "function", "function": {"name": "files.stat"}},
        ] * 9,
    )

    payload = provider._request_payload(request, temperature=0, stream=False)
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert any(name.startswith("terminal__") for name in names)


@pytest.mark.parametrize(
    "prompt",
    [
        "¿Cómo puedo revisar qué versión de Python tengo?",
        "¿Qué comando sirve para ver la versión de Python?",
    ],
)
def test_qwen35_terminal_information_does_not_retain_terminal_tools(
    tmp_path: Path, prompt: str
) -> None:
    settings = Settings(
        _env_file=None,
        workspace_root=tmp_path,
        local_runtime_profile_id=QWEN35_PROFILE_ID,
    )
    provider = QwenLocalProvider(settings)
    request = ModelRequest(
        task_type="agent",
        messages=[ModelMessage(role="user", content=prompt)],
        tools=[
            {"type": "function", "function": {"name": "terminal.run"}},
            {"type": "function", "function": {"name": "files.stat"}},
        ] * 9,
    )

    payload = provider._request_payload(request, temperature=0, stream=False)
    names = [tool["function"]["name"] for tool in payload["tools"]]
    assert not any(name.startswith("terminal__") for name in names)
