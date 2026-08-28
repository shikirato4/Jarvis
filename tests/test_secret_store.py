from __future__ import annotations

import json

import pytest

from jarvis.security_runtime.secrets import (
    EnvironmentSecretStore,
    KeyringSecretStore,
    NvidiaSecretManager,
    SecretStoreUnavailable,
)


SECRET = "nvapi-sentinel-never-expose"


class FakeKeyringBackend:
    priority = 1

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


def test_secret_manager_prefers_operating_system_store_and_exposes_only_status() -> (
    None
):
    backend = FakeKeyringBackend()
    backend.set_password("jarvis", "nvidia-api-key", SECRET)
    manager = NvidiaSecretManager(
        secure_store=KeyringSecretStore(backend=backend),
        environment_store=EnvironmentSecretStore(
            environ={"NVIDIA_API_KEY": "environment-value"}
        ),
    )

    resolved = manager.resolve()
    status = manager.status()

    assert resolved is not None
    assert resolved.reveal() == SECRET
    assert SECRET not in repr(resolved)
    assert status.configured is True
    assert status.source == "os_keyring"
    assert SECRET not in json.dumps(status.model_dump())


def test_secret_manager_saves_replaces_and_deletes_only_in_secure_store() -> None:
    backend = FakeKeyringBackend()
    manager = NvidiaSecretManager(
        secure_store=KeyringSecretStore(backend=backend),
        environment_store=EnvironmentSecretStore(environ={}),
    )

    assert manager.save(SECRET).configured is True
    assert manager.resolve().reveal() == SECRET  # type: ignore[union-attr]
    assert manager.save("replacement").configured is True
    assert manager.resolve().reveal() == "replacement"  # type: ignore[union-attr]
    assert manager.delete().configured is False
    assert manager.resolve() is None


def test_secret_manager_falls_back_to_environment_without_making_it_writable() -> None:
    manager = NvidiaSecretManager(
        secure_store=None,
        environment_store=EnvironmentSecretStore(environ={"NVIDIA_API_KEY": SECRET}),
    )

    assert manager.resolve().reveal() == SECRET  # type: ignore[union-attr]
    assert manager.status().source == "environment"
    with pytest.raises(SecretStoreUnavailable, match="almacén seguro"):
        manager.save("replacement")


def test_empty_secret_is_rejected_without_echoing_it() -> None:
    manager = NvidiaSecretManager(
        secure_store=KeyringSecretStore(backend=FakeKeyringBackend()),
        environment_store=EnvironmentSecretStore(environ={}),
    )

    with pytest.raises(ValueError, match="vacía"):
        manager.save("   ")
