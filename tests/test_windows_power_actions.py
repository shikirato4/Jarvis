from __future__ import annotations

from jarvis.actions.registry import ActionRegistry
from jarvis.action_executor.security import (
    SecurityDecisionStatus,
    TypedActionSecurityGate,
)
from jarvis.action_executor.models import ActionRequest
from jarvis.modules.windows_module import WindowsModule
from jarvis.windows_runtime import (
    InMemoryWindowsControlBackend,
    WindowsRuntimeService,
)


def _service() -> WindowsRuntimeService:
    return WindowsRuntimeService(InMemoryWindowsControlBackend())


def test_lock_workstation_runs_directly_and_is_recorded() -> None:
    service = _service()

    observation = service.lock_workstation()

    assert observation.operation == "lock_workstation"
    assert observation.verified is True


def test_power_operation_is_recorded_with_name() -> None:
    service = _service()

    for operation in ("suspend", "shutdown", "restart"):
        observation = service.power(operation)
        assert observation.operation == f"power_{operation}"
        assert observation.verified is True

    recorded = service.backend.power_operations  # noqa: SLF001
    assert recorded == ["suspend", "shutdown", "restart"]


def test_power_rejects_unknown_operations() -> None:
    service = _service()

    try:
        service.power("explode")
    except Exception as exc:  # noqa: BLE001 - asserting the guard itself
        assert "unsupported" in str(exc).casefold() or "not supported" in str(exc).casefold()
    else:
        raise AssertionError("unknown power operation must be rejected")


def test_windows_module_registers_workstation_and_power_actions() -> None:
    registry = ActionRegistry()
    WindowsModule(_service()).register_actions(registry)

    lock = registry.get("windows.workstation.lock")
    assert lock is not None
    assert lock.risk.value == "safe"

    for name in ("windows.power.suspend", "windows.power.shutdown", "windows.power.restart"):
        definition = registry.get(name)
        assert definition is not None, name
        assert definition.risk.value == "sensitive", name


def test_security_gate_allows_lock_but_confirms_power() -> None:
    registry = ActionRegistry()
    WindowsModule(_service()).register_actions(registry)
    gate = TypedActionSecurityGate()

    lock_decision = gate.evaluate(
        registry.get("windows.workstation.lock"),
        ActionRequest(run_id="r", call_id="c1", action="windows.workstation.lock", arguments={}),
    )
    assert lock_decision.status is SecurityDecisionStatus.ALLOWED

    power_decision = gate.evaluate(
        registry.get("windows.power.suspend"),
        ActionRequest(run_id="r", call_id="c2", action="windows.power.suspend", arguments={}),
    )
    assert power_decision.status is SecurityDecisionStatus.CONFIRMATION_REQUIRED
