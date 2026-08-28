from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
from threading import RLock
from typing import Any, Protocol

from jarvis.models.base import JarvisBaseModel


class SecretStoreUnavailable(RuntimeError):
    """Raised when a write requires an unavailable OS-backed secret store."""


@dataclass(frozen=True)
class ResolvedSecret:
    _value: str = field(repr=False)
    source: str

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"ResolvedSecret(source={self.source!r}, value='**********')"

    def __str__(self) -> str:
        return "**********"


class SecretStatus(JarvisBaseModel):
    configured: bool
    source: str
    validity: str = "unknown"


class SecretStore(Protocol):
    source_name: str

    def get(self) -> str | None: ...


class WritableSecretStore(SecretStore, Protocol):
    def set(self, value: str) -> None: ...

    def delete(self) -> None: ...


class EnvironmentSecretStore:
    source_name = "environment"

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        variable_name: str = "NVIDIA_API_KEY",
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._variable_name = variable_name

    def get(self) -> str | None:
        value = self._environ.get(self._variable_name, "").strip()
        return value or None


class KeyringSecretStore:
    source_name = "os_keyring"

    def __init__(
        self,
        *,
        backend: Any,
        service_name: str = "jarvis",
        username: str = "nvidia-api-key",
    ) -> None:
        self._backend = backend
        self._service_name = service_name
        self._username = username

    @classmethod
    def from_system(cls) -> KeyringSecretStore:
        try:
            import keyring
        except ImportError as exc:
            raise SecretStoreUnavailable(
                "El almacén seguro del sistema no está instalado; instala el extra desktop o configura NVIDIA_API_KEY en .env."
            ) from exc
        backend = keyring.get_keyring()
        if float(getattr(backend, "priority", 0)) <= 0:
            raise SecretStoreUnavailable(
                "El sistema no ofrece un backend keyring utilizable; configura NVIDIA_API_KEY en .env."
            )
        return cls(backend=backend)

    def get(self) -> str | None:
        value = self._backend.get_password(self._service_name, self._username)
        return value.strip() if isinstance(value, str) and value.strip() else None

    def set(self, value: str) -> None:
        self._backend.set_password(self._service_name, self._username, value)

    def delete(self) -> None:
        self._backend.delete_password(self._service_name, self._username)


class NvidiaSecretManager:
    def __init__(
        self,
        *,
        secure_store: WritableSecretStore | None,
        environment_store: SecretStore | None = None,
    ) -> None:
        self._secure_store = secure_store
        self._environment_store = environment_store or EnvironmentSecretStore()
        self._validity = "unknown"
        self._lock = RLock()

    @classmethod
    def from_system(cls, *, fallback_value: str | None = None) -> NvidiaSecretManager:
        try:
            secure_store: WritableSecretStore | None = KeyringSecretStore.from_system()
        except SecretStoreUnavailable:
            secure_store = None
        fallback = {"NVIDIA_API_KEY": fallback_value} if fallback_value else {}
        return cls(
            secure_store=secure_store,
            environment_store=EnvironmentSecretStore(environ=fallback),
        )

    def resolve(self) -> ResolvedSecret | None:
        if self._secure_store is not None:
            secure_value = self._secure_store.get()
            if secure_value:
                return ResolvedSecret(secure_value, self._secure_store.source_name)
        environment_value = self._environment_store.get()
        if environment_value:
            return ResolvedSecret(
                environment_value, self._environment_store.source_name
            )
        return None

    def status(self) -> SecretStatus:
        resolved = self.resolve()
        with self._lock:
            validity = self._validity
        return SecretStatus(
            configured=resolved is not None,
            source=resolved.source if resolved is not None else "none",
            validity=validity,
        )

    def mark_validity(self, validity: str) -> None:
        if validity not in {"unknown", "valid", "invalid"}:
            raise ValueError("Estado de validez de clave no reconocido.")
        with self._lock:
            self._validity = validity

    def save(self, value: str) -> SecretStatus:
        normalized = value.strip()
        if not normalized:
            raise ValueError("La clave NVIDIA no puede estar vacía.")
        if self._secure_store is None:
            raise SecretStoreUnavailable(
                "No hay un almacén seguro disponible; configura la clave en el archivo .env local."
            )
        self._secure_store.set(normalized)
        self.mark_validity("unknown")
        return self.status()

    def delete(self) -> SecretStatus:
        if self._secure_store is not None and self._secure_store.get() is not None:
            self._secure_store.delete()
        self.mark_validity("unknown")
        return self.status()
