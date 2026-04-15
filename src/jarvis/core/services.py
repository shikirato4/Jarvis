from __future__ import annotations

from typing import Protocol

from .models import ServiceStatus


class RuntimeServiceContract(Protocol):
    service_name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def health(self) -> ServiceStatus: ...

    def status(self) -> dict[str, object]: ...
