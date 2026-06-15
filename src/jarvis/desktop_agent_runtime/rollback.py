from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import DesktopAgentStep, DesktopStepActionType


@dataclass(frozen=True)
class RollbackStep:
    action_type: str
    description: str
    risk: str = "medium"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "description": self.description,
            "risk": self.risk,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class RollbackPlan:
    rollback_available: bool
    rollback_risk: str
    rollback_description: str
    steps: list[RollbackStep] = field(default_factory=list)
    safe_to_auto_execute: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_available": self.rollback_available,
            "rollback_risk": self.rollback_risk,
            "rollback_description": self.rollback_description,
            "safe_to_auto_execute": self.safe_to_auto_execute,
            "steps": [step.to_dict() for step in self.steps],
        }


class RollbackPlanner:
    def for_step(self, step: DesktopAgentStep) -> RollbackPlan:
        path = str(step.payload.get("path") or "")
        destination = str(step.payload.get("destination_path") or "")
        new_name = str(step.payload.get("new_name") or "")
        if step.action_type == DesktopStepActionType.CREATE_FOLDER and path:
            return RollbackPlan(
                rollback_available=True,
                rollback_risk="medium",
                rollback_description="Se puede borrar la carpeta creada solo si sigue vacia.",
                steps=[
                    RollbackStep(
                        "delete_empty_folder",
                        "Borrar la carpeta creada si esta vacia.",
                        parameters={"path": path},
                    )
                ],
            )
        if step.action_type == DesktopStepActionType.CREATE_FILE and path:
            return RollbackPlan(
                rollback_available=True,
                rollback_risk="medium",
                rollback_description="Se puede borrar el archivo creado si no fue usado para otros cambios.",
                steps=[RollbackStep("delete_file", "Borrar el archivo creado.", parameters={"path": path})],
            )
        if step.action_type == DesktopStepActionType.COPY_FILE and destination:
            return RollbackPlan(
                rollback_available=True,
                rollback_risk="medium",
                rollback_description="Se puede borrar la copia creada si no fue modificada despues.",
                steps=[RollbackStep("delete_copied_file", "Borrar la copia creada.", parameters={"path": destination})],
            )
        if step.action_type == DesktopStepActionType.MOVE_FILE and destination:
            return RollbackPlan(
                rollback_available=True,
                rollback_risk="medium",
                rollback_description="Se puede mover el archivo de vuelta si el origen original sigue disponible como referencia.",
                steps=[RollbackStep("move_back", "Mover el archivo de vuelta.", parameters={"destination_path": destination})],
            )
        if step.action_type == DesktopStepActionType.RENAME_FILE and new_name:
            return RollbackPlan(
                rollback_available=True,
                rollback_risk="medium",
                rollback_description="Se puede renombrar de vuelta si no existe conflicto de nombres.",
                steps=[RollbackStep("rename_back", "Restaurar el nombre anterior.", parameters={"new_name": new_name})],
            )
        return RollbackPlan(
            rollback_available=False,
            rollback_risk="unknown",
            rollback_description="Rollback no garantizado para esta accion.",
        )
