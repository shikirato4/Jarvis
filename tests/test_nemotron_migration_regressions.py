from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import ConfigurationError
from jarvis.core.events import EventBus
from jarvis.core.modes import ModeManager
from jarvis.desktop_runtime.chat import DesktopChatEngine
from jarvis.models_runtime.catalog import build_default_model_catalog
from jarvis.models_runtime.coordinator import PersistentModeStore
from jarvis.models_runtime.hybrid import LLMMode
from jarvis.models_runtime.registry import ProviderRegistry
from jarvis.models_runtime.router import ModelRouter
from jarvis.models_runtime.service import ModelService
from jarvis.persistent_config import (
    PersistentConfig,
    PersistentConfigWriteError,
    get_config_path,
    load_persistent_config,
    save_persistent_config,
)


REMOTE_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
LOCAL_MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
LEGACY_PROVIDER = "oll" + "ama"
LEGACY_MODEL = "gpt" + "-oss:20b"


def test_application_registers_only_nemotron_chat_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "jarvis.models_runtime.coordinator.get_config_path",
        lambda _root=None: tmp_path / "isolated-mode.json",
    )
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        llm_mode="auto",
        embeddings_enabled=False,
    )

    application = build_application(settings)

    assert [
        provider.provider_name
        for provider in application.provider_registry.list_providers()
    ] == ["nemotron_omni_local", "nvidia_nim"]
    assert application.model_service.provider_selection().selected_mode.value == "auto"


def test_model_service_rejects_construction_without_provider_coordinator() -> None:
    settings = Settings(_env_file=None)
    catalog = build_default_model_catalog(settings)
    registry = ProviderRegistry()
    mode_manager = ModeManager(settings.default_mode)

    with pytest.raises(ConfigurationError, match="ProviderCoordinator"):
        ModelService(
            settings,
            mode_manager,
            registry,
            catalog,
            ModelRouter(catalog, mode_manager),
            EventBus(),
            coordinator=None,
        )


def test_legacy_configuration_migrates_to_auto_without_writing_on_read(
    tmp_path: Path,
) -> None:
    config_path = get_config_path(tmp_path)
    config_path.parent.mkdir(parents=True)
    legacy_payload = {
        "preferred_mode": "offline",
        "local_provider": LEGACY_PROVIDER,
        "local_base_url": "http://127.0.0.1:11434",
        "local_model": LEGACY_MODEL,
        "allow_online_for_public_tasks": True,
        "unrelated_user_setting": "keep-me",
    }
    original = json.dumps(legacy_payload, indent=2)
    config_path.write_text(original, encoding="utf-8")

    first = load_persistent_config(tmp_path)
    second = load_persistent_config(tmp_path)

    assert first == second
    assert first.preferred_mode == "auto"
    assert first.local_provider == "nemotron_omni_local"
    assert first.local_base_url == "http://127.0.0.1:8000/v1"
    assert first.local_model == LOCAL_MODEL
    assert config_path.read_text(encoding="utf-8") == original

    save_persistent_config(first, tmp_path)
    saved = config_path.read_text(encoding="utf-8")
    assert LEGACY_PROVIDER not in saved.casefold()
    assert "gpt" + "-oss" not in saved.casefold()
    assert load_persistent_config(tmp_path) == first


def test_missing_persistent_config_is_side_effect_free(tmp_path: Path) -> None:
    config_path = get_config_path(tmp_path)

    assert load_persistent_config(tmp_path) == PersistentConfig.defaults()
    assert not config_path.exists()


def test_default_mode_store_reads_workspace_legacy_config_without_overwriting_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    user_config = tmp_path / "user-config"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("jarvis.persistent_config.user_config_dir", lambda: user_config)
    legacy_path = tmp_path / "runtime" / "jarvis_config.json"
    legacy_path.parent.mkdir(parents=True)
    original = json.dumps(
        {
            "preferred_mode": "offline",
            "local_provider": LEGACY_PROVIDER,
            "local_model": LEGACY_MODEL,
        }
    )
    legacy_path.write_text(original, encoding="utf-8")

    store = PersistentModeStore(default_mode=LLMMode.LOCAL)

    assert store.load() is LLMMode.AUTO
    assert legacy_path.read_text(encoding="utf-8") == original
    assert not (user_config / "jarvis_config.json").exists()


def test_atomic_persistence_failure_is_reported_without_truncating_existing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PersistentConfig(preferred_mode="auto")
    save_persistent_config(config, tmp_path)
    config_path = get_config_path(tmp_path)
    original = config_path.read_bytes()
    monkeypatch.setattr(
        "jarvis.persistent_config.os.replace",
        lambda *_args: (_ for _ in ()).throw(PermissionError("locked")),
    )

    config.preferred_mode = "local"
    with pytest.raises(PersistentConfigWriteError):
        save_persistent_config(config, tmp_path)

    assert config_path.read_bytes() == original


class _FailingRuntime:
    def infer_model(self, _request):
        raise RuntimeError("connection refused")


class _Panels:
    @staticmethod
    def compose():
        return None


class _Actions:
    pass


def test_desktop_chat_provider_failure_never_reports_legacy_model_names() -> None:
    bridge = SimpleNamespace(runtime=_FailingRuntime())
    engine = DesktopChatEngine(bridge, _Panels(), _Actions(), Settings(_env_file=None))

    response = engine.handle("hola Jarvis")

    normalized = response.message.content.casefold()
    assert LEGACY_PROVIDER not in normalized
    assert "gpt" + "-oss" not in normalized
    assert "nemotron" in normalized
    assert response.raw_result["error"]["type"] == "RuntimeError"
