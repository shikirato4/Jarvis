from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .errors import ConfigurationError

if TYPE_CHECKING:
    from jarvis.actions.registry import ActionRegistry
    from jarvis.core.capabilities import CapabilityRegistry
    from jarvis.tools.registry import ToolRegistry


class JarvisModule(Protocol):
    name: str
    description: str

    def register_actions(self, registry: "ActionRegistry") -> None: ...

    def register_tools(self, registry: "ToolRegistry") -> None: ...

    def register_capabilities(self, registry: "CapabilityRegistry") -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(slots=True)
class ModuleRecord:
    name: str
    description: str


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: list[JarvisModule] = []

    @property
    def descriptors(self) -> list[ModuleRecord]:
        return [ModuleRecord(name=module.name, description=module.description) for module in self._modules]

    def register(self, module: JarvisModule) -> None:
        if any(item.name == module.name for item in self._modules):
            raise ConfigurationError(f"module '{module.name}' is already registered")
        self._modules.append(module)

    def register_actions(self, registry: "ActionRegistry") -> None:
        for module in self._modules:
            module.register_actions(registry)

    def register_tools(self, registry: "ToolRegistry") -> None:
        for module in self._modules:
            registrar = getattr(module, "register_tools", None)
            if callable(registrar):
                registrar(registry)

    def register_capabilities(self, registry: "CapabilityRegistry") -> None:
        for module in self._modules:
            registrar = getattr(module, "register_capabilities", None)
            if callable(registrar):
                registrar(registry)

    def start_all(self) -> None:
        for module in self._modules:
            module.start()

    def stop_all(self) -> None:
        for module in reversed(self._modules):
            module.stop()
