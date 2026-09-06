from jarvis.config import Settings
from jarvis.persistent_config import (
    PersistentConfig,
    load_persistent_config,
    save_persistent_config,
)
from jarvis.reasoning import ReasoningPreference, resolve_reasoning_preference


def test_reasoning_preference_is_backend_agnostic_and_auto_is_deterministic() -> None:
    assert (
        resolve_reasoning_preference(ReasoningPreference.OFF, "debug this")
        is ReasoningPreference.OFF
    )
    assert (
        resolve_reasoning_preference("auto", "debug this") is ReasoningPreference.XHIGH
    )
    assert resolve_reasoning_preference("auto", "hola") is ReasoningPreference.LOW
    assert Settings().reasoning_preference is ReasoningPreference.AUTO


def test_reasoning_preference_persists_without_model_backend(tmp_path) -> None:
    config = PersistentConfig(reasoning_preference="xhigh")
    save_persistent_config(config, tmp_path)
    assert load_persistent_config(tmp_path).reasoning_preference == "xhigh"
