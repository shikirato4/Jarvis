from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agent_mode import AgentAction, AgentMode, AgentPermissionMode, AgentRisk, AgentSafetyGate


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    category: str
    risk: str
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    outputs_schema: dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    supports_dry_run: bool = True
    supports_verify: bool = True
    supports_rollback: bool = False
    action_type: str = "agent_action"

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "skill": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "payload": payload,
            "requires_confirmation": self.requires_confirmation,
        }

    def dry_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "dry_run",
            "skill": self.name,
            "would_execute": False,
            "plan": self.plan(payload),
        }

    def execute(self, payload: dict[str, Any], *, executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
        if executor is None:
            return {"status": "blocked", "skill": self.name, "message": "Skill execution requires a runtime executor."}
        return executor(payload)

    def verify(self, result: dict[str, Any]) -> dict[str, Any]:
        return {"skill": self.name, "verified": bool(result.get("success", result.get("ok", False))), "result": result}

    def rollback(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.supports_rollback:
            return {"status": "unavailable", "skill": self.name, "message": "Rollback no seguro para esta skill."}
        return {"status": "planned", "skill": self.name, "payload": payload}


class AgentSkillRegistry:
    def __init__(self, skills: list[AgentSkill] | None = None) -> None:
        self._skills: dict[str, AgentSkill] = {}
        for skill in skills or builtin_agent_skills():
            self.register(skill)

    def register(self, skill: AgentSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"duplicate Agent Mode skill id: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> AgentSkill | None:
        return self._skills.get(name)

    def list(self) -> list[AgentSkill]:
        return list(self._skills.values())

    def authorize(
        self,
        skill_name: str,
        payload: dict[str, Any],
        *,
        safety_gate: AgentSafetyGate | None = None,
        agent_mode: AgentMode = AgentMode.GUIDED_CONTROL,
        permission_mode: AgentPermissionMode = AgentPermissionMode.NORMAL,
        confirmed: bool = False,
        strong_confirmed: bool = False,
        pin_verified: bool = False,
    ) -> dict[str, Any]:
        skill = self._skills[skill_name]
        gate = safety_gate or AgentSafetyGate()
        action = AgentAction(
            action_type=skill.action_type,
            title=skill.name,
            description=skill.description,
            payload=payload,
            risk=AgentRisk(str(skill.risk)),
            dry_run_supported=skill.supports_dry_run,
            verify_supported=skill.supports_verify,
            rollback_supported=skill.supports_rollback,
        )
        result = gate.authorize(
            action,
            mode=agent_mode,
            permission_mode=permission_mode,
            confirmed=confirmed,
            strong_confirmed=strong_confirmed,
            pin_verified=pin_verified,
        )
        return {
            "skill": skill.name,
            "decision": result.decision.value,
            "risk": result.risk.value,
            "reason": result.reason,
            "requires_pin": result.requires_pin,
        }


def builtin_agent_skills() -> list[AgentSkill]:
    return [
        AgentSkill("inspect_screen", "Inspectar pantalla con vision segura.", "eyes", "low", action_type="inspect_screen"),
        AgentSkill("inspect_active_window", "Inspectar ventana activa con fallback seguro.", "eyes", "low", action_type="inspect_active_window"),
        AgentSkill("browser_search", "Buscar con Brave Search como fuente web.", "browser", "low", action_type="browser_search"),
        AgentSkill("open_url", "Abrir URL o carpeta segura.", "browser", "low", action_type="open_url"),
        AgentSkill("github_repo_learning", "Buscar y resumir repos publicos sin ejecutar codigo.", "learning", "low", action_type="browser_search"),
        AgentSkill("rank_repositories", "Rankear repos por relevancia, calidad y riesgo.", "learning", "low", action_type="browser_search"),
        AgentSkill("summarize_docs", "Resumir README/docs seguros.", "learning", "low", action_type="browser_search"),
        AgentSkill("create_folder", "Crear carpeta con confirmacion.", "files", "medium", requires_confirmation=True, action_type="create_folder", supports_rollback=True),
        AgentSkill("create_file", "Crear archivo con confirmacion.", "files", "medium", requires_confirmation=True, action_type="create_file", supports_rollback=True),
        AgentSkill("copy_file", "Copiar archivo con confirmacion.", "files", "medium", requires_confirmation=True, action_type="copy_file", supports_rollback=True),
        AgentSkill("move_file", "Mover archivo con confirmacion.", "files", "medium", requires_confirmation=True, action_type="move_file", supports_rollback=True),
        AgentSkill("rename_file", "Renombrar archivo con confirmacion.", "files", "medium", requires_confirmation=True, action_type="rename_file", supports_rollback=True),
        AgentSkill("verify_file_exists", "Verificar existencia de archivo o carpeta.", "files", "low", action_type="search_file"),
        AgentSkill("safe_download_preview", "Preparar descarga sin ejecutarla.", "downloads", "medium", requires_confirmation=True, action_type="download_file"),
        AgentSkill("safe_install_preview", "Preparar instalacion con confirmacion fuerte.", "downloads", "high", requires_confirmation=True, action_type="open_installer"),
        AgentSkill("generate_local_image", "Generar imagen local con JuggernautXL sin Fooocus.", "image", "low", action_type="generate_local_image"),
        AgentSkill("improve_image_prompt", "Mejorar prompt visual sin ejecutar generacion.", "image", "low", action_type="inspect_screen"),
        AgentSkill("open_image_output_folder", "Abrir carpeta local de outputs de imagen.", "image", "low", action_type="open_folder"),
        AgentSkill("cancel_image_generation", "Cancelar generacion de imagen en curso.", "image", "low", action_type="agent_action"),
        AgentSkill("unload_image_model", "Descargar modelo de imagen de memoria.", "image", "medium", requires_confirmation=True, action_type="modify_setting"),
        AgentSkill("rollback_action", "Planear rollback cuando sea seguro.", "rollback", "medium", requires_confirmation=True, action_type="rollback_action"),
        AgentSkill("edit_project_with_code_agent", "Delegar edicion de codigo al Code Agent.", "code", "medium", requires_confirmation=True, action_type="modify_setting"),
    ]
