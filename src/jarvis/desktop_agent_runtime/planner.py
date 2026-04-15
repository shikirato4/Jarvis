from __future__ import annotations

import json
import re
from uuid import uuid4

from jarvis.models_runtime.base import ModelMessage, ModelRequest

from .models import (
    DesktopAgentExpectation,
    DesktopAgentModelDecision,
    DesktopAgentModelSuggestion,
    DesktopAgentPlan,
    DesktopAgentRiskLevel,
    DesktopAgentStep,
    DesktopAgentVerificationResult,
    DesktopStepActionType,
    DesktopWorldState,
)


class DesktopAgentPlanner:
    def __init__(self, settings, runtime=None, logger=None) -> None:
        self._settings = settings
        self._runtime = runtime
        self._logger = logger

    def plan(self, world: DesktopWorldState) -> DesktopAgentPlan:
        goal = world.current_goal.strip()
        lowered = goal.casefold()
        steps: list[DesktopAgentStep]
        strategy = "grounded_generic"
        if self._is_browser_search(lowered):
            strategy = "grounded_browser_search"
            app, query = self._extract_browser_search(goal)
            world.target_application = app
            steps = self._browser_search_steps(app, query, world)
        elif self._is_open_and_type(lowered):
            strategy = "grounded_open_and_type"
            app = self._extract_application(goal)
            text = self._extract_literal_tail(goal)
            world.target_application = app
            steps = [
                self._step(
                    "open-app",
                    f"Abrir {app}",
                    DesktopStepActionType.OPEN_APPLICATION,
                    f"Abrir {app}.",
                    {"application": app},
                    DesktopAgentExpectation(active_window_contains=app, process_name_contains=self._infer_process_hint(app)),
                    subgoal=f"tener {app} disponible",
                    success_label=f"{app} abierto",
                    alternatives=["refocus existing window", "search same process window"],
                ),
                self._step(
                    "focus-app",
                    f"Enfocar {app}",
                    DesktopStepActionType.FOCUS_WINDOW,
                    f"Enfocar {app}.",
                    {"target_window": app},
                    DesktopAgentExpectation(active_window_contains=app, process_name_contains=self._infer_process_hint(app)),
                    subgoal=f"confirmar que {app} es la ventana destino",
                    success_label=f"{app} activo",
                    alternatives=["buscar otra ventana del mismo proceso"],
                ),
                self._step(
                    "write-text",
                    "Escribir texto",
                    DesktopStepActionType.WRITE_TEXT,
                    "Escribir el texto solicitado.",
                    {"text": text, "target_window": app},
                    DesktopAgentExpectation(
                        active_window_contains=app,
                        visible_text_contains=[text[:60]] if text else [],
                        required_context_signals=["window_title:" + app.casefold()],
                    ),
                    subgoal="insertar el texto solicitado en la app objetivo",
                    success_label="texto visible tras escribir",
                    alternatives=["reobserve and retry", "refocus before write"],
                ),
            ]
        elif self._is_write_to_active_window(lowered):
            strategy = "grounded_active_window_write"
            text = self._extract_literal_tail(goal)
            active_title = world.active_window.title if world.active_window else ""
            world.target_window_title = active_title or None
            steps = [
                self._step(
                    "focus-active",
                    "Usar ventana activa",
                    DesktopStepActionType.FOCUS_WINDOW,
                    "Mantener la ventana activa como destino.",
                    {"target_window": active_title},
                    DesktopAgentExpectation(active_window_contains=active_title or None),
                    subgoal="preservar el foco en la ventana activa",
                    success_label="ventana activa confirmada",
                    alternatives=["re-observe current window"],
                ),
                self._step(
                    "write-active",
                    "Escribir texto",
                    DesktopStepActionType.WRITE_TEXT,
                    "Escribir el texto solicitado.",
                    {"text": text, "target_window": active_title},
                    DesktopAgentExpectation(
                        active_window_contains=active_title or None,
                        visible_text_contains=[text[:60]] if text else [],
                    ),
                    subgoal="insertar el texto en la ventana activa",
                    success_label="texto insertado en ventana activa",
                    alternatives=["refocus", "retry with selection evidence"],
                ),
            ]
        elif self._is_file_search_and_open(lowered):
            strategy = "grounded_file_search"
            query = self._extract_file_query(goal)
            steps = [
                self._step(
                    "search-file",
                    "Buscar archivo",
                    DesktopStepActionType.SEARCH_FILE,
                    f"Buscar archivo '{query}'.",
                    {"query": query},
                    DesktopAgentExpectation(search_results_min=1),
                    subgoal="encontrar una ruta segura que coincida con el archivo pedido",
                    success_label="archivo localizado",
                    alternatives=["broaden search"],
                ),
                self._step(
                    "open-file",
                    "Abrir archivo",
                    DesktopStepActionType.OPEN_PATH,
                    "Abrir el primer resultado seguro.",
                    {"result_from": "search-file"},
                    DesktopAgentExpectation(action_success_required=True),
                    risk=DesktopAgentRiskLevel.MEDIUM,
                    subgoal="abrir el archivo encontrado sin salir del flujo seguro",
                    success_label="archivo abierto",
                    alternatives=["abort safely if path is ambiguous"],
                ),
            ]
        elif self._is_word_continue(lowered):
            strategy = "grounded_word_writing"
            text = self._extract_literal_tail(goal)
            world.target_application = "word"
            steps = [
                self._step(
                    "open-word",
                    "Abrir Word",
                    DesktopStepActionType.OPEN_APPLICATION,
                    "Abrir Microsoft Word.",
                    {"application": "word"},
                    DesktopAgentExpectation(active_window_contains="word", process_name_contains="winword"),
                    subgoal="tener Word listo",
                    success_label="Word abierto",
                    alternatives=["refocus existing Word window"],
                ),
                self._step(
                    "focus-word",
                    "Enfocar Word",
                    DesktopStepActionType.FOCUS_WINDOW,
                    "Enfocar la ventana de Word.",
                    {"target_window": "word"},
                    DesktopAgentExpectation(active_window_contains="word", process_name_contains="winword", required_context_signals=["word_document_active"]),
                    subgoal="confirmar documento activo en Word",
                    success_label="documento Word en foco",
                    alternatives=["buscar otra ventana WINWORD.EXE"],
                ),
                self._step(
                    "analyze-word",
                    "Detectar documento",
                    DesktopStepActionType.WRITING_ANALYZE,
                    "Analizar el contexto del documento actual.",
                    {"prompt": goal, "target_window": "Word"},
                    DesktopAgentExpectation(active_window_contains="word", required_context_signals=["word_document_active"]),
                    subgoal="entender el documento antes de escribir",
                    success_label="contexto del documento leído",
                    alternatives=["reobserve document window"],
                ),
                self._step(
                    "continue-word",
                    "Continuar escritura",
                    DesktopStepActionType.WRITING_CONTINUE,
                    "Continuar o insertar texto en el documento.",
                    {"prompt": goal, "target_window": "Word", "text": text},
                    DesktopAgentExpectation(active_window_contains="word", visible_text_contains=[text[:60]] if text else []),
                    subgoal="insertar o continuar el texto solicitado en Word",
                    success_label="texto añadido al documento",
                    alternatives=["refocus before write", "re-analyze document"],
                ),
            ]
        else:
            application = self._extract_application(goal)
            strategy = "grounded_open_application"
            world.target_application = application
            steps = [
                self._step(
                    "open-app",
                    f"Abrir {application}",
                    DesktopStepActionType.OPEN_APPLICATION,
                    f"Abrir {application}.",
                    {"application": application},
                    DesktopAgentExpectation(active_window_contains=application, process_name_contains=self._infer_process_hint(application)),
                    subgoal=f"abrir {application}",
                    success_label=f"{application} abierto",
                    alternatives=["refocus existing window"],
                ),
                self._step(
                    "focus-app",
                    f"Enfocar {application}",
                    DesktopStepActionType.FOCUS_WINDOW,
                    f"Enfocar {application}.",
                    {"target_window": application},
                    DesktopAgentExpectation(active_window_contains=application, process_name_contains=self._infer_process_hint(application)),
                    subgoal=f"confirmar foco en {application}",
                    success_label=f"{application} activo",
                    alternatives=["find another matching window"],
                ),
            ]
        return DesktopAgentPlan(
            mission_id=world.mission_id,
            strategy=strategy,
            steps=steps,
            metadata={
                "goal": goal,
                "active_window": world.active_window.title if world.active_window else None,
                "target_application": world.target_application,
                "source": "heuristic",
            },
        )

    def replan(
        self,
        world: DesktopWorldState,
        *,
        reason: str,
        failed_step: DesktopAgentStep | None = None,
        verification: DesktopAgentVerificationResult | None = None,
    ) -> DesktopAgentPlan:
        suggestion = self.propose_replan(world, reason=reason, failed_step=failed_step, verification=verification)
        if suggestion is not None and suggestion.decision == DesktopAgentModelDecision.REPLAN and suggestion.steps:
            return DesktopAgentPlan(
                mission_id=world.mission_id,
                strategy=suggestion.strategy,
                steps=suggestion.steps,
                metadata={
                    "goal": world.current_goal,
                    "replan_reason": reason,
                    "completed_steps": list(world.completed_steps),
                    "last_observation_summary": world.last_observation_summary,
                    "source": "model",
                    "model_rationale": suggestion.rationale,
                    **suggestion.metadata,
                },
            )
        plan = self.plan(world)
        plan.metadata["replan_reason"] = reason
        plan.metadata["completed_steps"] = list(world.completed_steps)
        plan.metadata["last_observation_summary"] = world.last_observation_summary
        plan.metadata["source"] = "heuristic_fallback"
        return plan

    def propose_replan(
        self,
        world: DesktopWorldState,
        *,
        reason: str,
        failed_step: DesktopAgentStep | None,
        verification: DesktopAgentVerificationResult | None,
    ) -> DesktopAgentModelSuggestion | None:
        if self._runtime is None:
            return None
        try:
            prompt = self._build_replan_prompt(world, reason=reason, failed_step=failed_step, verification=verification)
            response = self._runtime.infer_model(
                ModelRequest(
                    task_type="planning",
                    logical_model="planner",
                    required_capabilities=("planning", "reasoning"),
                    temperature=0.1,
                    messages=prompt,
                    metadata={"component": "desktop_agent_replanner", "mission_id": world.mission_id},
                )
            )
            return self._parse_model_suggestion(world, response.content)
        except Exception as exc:  # noqa: BLE001
            if self._logger is not None:
                self._logger.warning("desktop_agent_model_replan_failed", extra={"error": str(exc), "mission_id": world.mission_id})
            return None

    def _build_replan_prompt(
        self,
        world: DesktopWorldState,
        *,
        reason: str,
        failed_step: DesktopAgentStep | None,
        verification: DesktopAgentVerificationResult | None,
    ) -> list[ModelMessage]:
        current_context = {
            "goal": world.current_goal,
            "current_subgoal": world.current_subgoal,
            "failed_step": failed_step.model_dump(mode="json") if failed_step is not None else None,
            "expected_next_state": world.expected_next_state,
            "verification": verification.model_dump(mode="json") if verification is not None else None,
            "reason": reason,
            "active_window": world.active_window.model_dump(mode="json") if world.active_window else None,
            "visible_text": world.visible_text[:500],
            "selection_text": world.selection_text[:200],
            "clipboard_text": world.clipboard_text[:200],
            "context_signals": world.context_signals,
            "detected_targets": [target.model_dump(mode="json") for target in world.detected_targets[:8]],
            "attempted_steps": world.memory.attempted_steps,
            "completed_steps": world.completed_steps,
            "previous_attempts": {
                "fallbacks": world.memory.attempted_fallbacks,
                "recovery_attempts_by_step": world.memory.recovery_attempts_by_step,
                "last_error": world.last_error,
                "successful_strategies": world.memory.successful_strategies,
            },
        }
        system = (
            "You are a desktop-agent replanning model. "
            "Decide how to continue after a failed UI step. "
            "Return JSON only. Never propose code execution, shell commands, policy bypass, destructive actions, or unsupported actions. "
            "You may only use these action_type values: "
            + ", ".join(item.value for item in DesktopStepActionType)
            + ". "
            "Prefer a small safe subplan. Avoid repeating the same failed strategy. "
            "Schema: "
            '{"decision":"replan|retry|abort","strategy":"short_strategy","rationale":"why","metadata":{"confidence":"low|medium|high"},"steps":[{"step_id":"id","title":"title","action_type":"focus_window","action":"desc","precondition":"...",'
            '"subgoal":"...","success_label":"...","fallback":"...","alternatives":["..."],"payload":{},"risk_level":"low|medium|high|critical","max_retries":1,"verification":{"active_window_contains":null,"process_name_contains":null,"visible_text_contains":[],"visible_text_not_contains":[],"required_context_signals":[],"forbidden_context_signals":[],"expected_targets":[],"selection_contains":[],"clipboard_contains":[],"search_results_min":null,"action_success_required":true}}]}. '
            'If no safe continuation exists, return {"decision":"abort","strategy":"safe_abort","rationale":"...","steps":[]}.'
        )
        return [
            ModelMessage(role="system", content=system),
            ModelMessage(role="user", content=json.dumps(current_context, ensure_ascii=False)),
        ]

    def _parse_model_suggestion(self, world: DesktopWorldState, content: str) -> DesktopAgentModelSuggestion | None:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            return None
        decision = DesktopAgentModelDecision(str(payload.get("decision", DesktopAgentModelDecision.REPLAN.value)))
        strategy = str(payload.get("strategy") or "model_replan")
        rationale = str(payload.get("rationale") or "Model proposed a continuation strategy.")
        raw_steps = payload.get("steps", [])
        steps: list[DesktopAgentStep] = []
        if isinstance(raw_steps, list):
            for index, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    continue
                try:
                    step = DesktopAgentStep.model_validate(
                        {
                            "step_id": f"model-{index}-{uuid4().hex[:6]}",
                            "title": item.get("title", f"Model step {index}"),
                            "action_type": item.get("action_type"),
                            "precondition": item.get("precondition", "desktop runtime available and observation refreshed"),
                            "action": item.get("action", item.get("title", f"Model step {index}")),
                            "verification": item.get("verification", {}),
                            "subgoal": item.get("subgoal"),
                            "success_label": item.get("success_label"),
                            "fallback": item.get("fallback", "re-observe and recover safely"),
                            "alternatives": item.get("alternatives", []),
                            "payload": item.get("payload", {}),
                            "risk_level": item.get("risk_level", DesktopAgentRiskLevel.LOW.value),
                            "max_retries": item.get("max_retries", 1),
                        }
                    )
                except Exception:
                    continue
                if self._is_duplicate_of_history(world, step):
                    continue
                steps.append(step)
        return DesktopAgentModelSuggestion(
            decision=decision,
            strategy=strategy,
            rationale=rationale,
            steps=steps,
            metadata=payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {},
        )

    @staticmethod
    def _is_duplicate_of_history(world: DesktopWorldState, step: DesktopAgentStep) -> bool:
        payload = json.dumps(step.payload, sort_keys=True, ensure_ascii=False)
        for action in world.recent_actions:
            if action.action_type == step.action_type.value and json.dumps(action.receipt.get("request_payload", action.receipt), sort_keys=True, ensure_ascii=False) == payload:
                return True
        for attempted in world.memory.attempted_steps:
            if attempted == step.step_id:
                return True
        return False

    def _browser_search_steps(self, app: str, query: str, world: DesktopWorldState) -> list[DesktopAgentStep]:
        browser_already_active = bool(world.active_window and app.casefold() in world.active_window.title.casefold())
        steps: list[DesktopAgentStep] = []
        if not browser_already_active:
            steps.append(
                self._step(
                    "open-browser",
                    "Abrir navegador",
                    DesktopStepActionType.OPEN_APPLICATION,
                    f"Abrir {app}.",
                    {"application": app},
                    DesktopAgentExpectation(active_window_contains=app, process_name_contains=self._infer_process_hint(app)),
                    subgoal=f"abrir {app}",
                    success_label=f"{app} abierto",
                    alternatives=["focus existing browser", "search same process window"],
                )
            )
        steps.extend(
            [
                self._step(
                    "focus-browser",
                    "Enfocar navegador",
                    DesktopStepActionType.FOCUS_WINDOW,
                    f"Enfocar {app}.",
                    {"target_window": app},
                    DesktopAgentExpectation(active_window_contains=app, process_name_contains=self._infer_process_hint(app), required_context_signals=["browser_active"]),
                    subgoal="asegurar que el navegador correcto esta activo",
                    success_label="navegador en foco",
                    alternatives=["search another browser window"],
                ),
                self._step(
                    "focus-address-bar",
                    "Enfocar barra",
                    DesktopStepActionType.HOTKEY,
                    "Llevar el foco a la barra de direcciones o busqueda.",
                    {"keys": ("ctrl", "l"), "target_window": app},
                    DesktopAgentExpectation(active_window_contains=app, required_context_signals=["browser_active"]),
                    subgoal="llevar el foco al campo de busqueda",
                    success_label="foco de busqueda preparado",
                    alternatives=["retry with explicit browser refocus"],
                ),
                self._step(
                    "type-query",
                    "Escribir busqueda",
                    DesktopStepActionType.WRITE_TEXT,
                    f"Escribir '{query}'.",
                    {"text": query, "target_window": app},
                    DesktopAgentExpectation(
                        active_window_contains=app,
                        visible_text_contains=[query],
                    ),
                    subgoal="colocar la consulta en el campo visible",
                    success_label="consulta visible",
                    alternatives=["reobserve and retry", "refocus before write"],
                ),
                self._step(
                    "submit-query",
                    "Enviar busqueda",
                    DesktopStepActionType.HOTKEY,
                    "Enviar la busqueda.",
                    {"keys": ("enter",), "target_window": app},
                    DesktopAgentExpectation(
                        active_window_contains=app,
                        visible_text_contains=[query],
                        required_context_signals=["browser_active"],
                    ),
                    subgoal="ejecutar la busqueda y mantener evidencia de contexto",
                    success_label="busqueda enviada",
                    alternatives=["retry after fresh observation", "replan if UI changed"],
                ),
            ]
        )
        return steps

    def _step(
        self,
        step_id: str,
        title: str,
        action_type: DesktopStepActionType,
        action: str,
        payload: dict,
        verification: DesktopAgentExpectation,
        *,
        risk: DesktopAgentRiskLevel = DesktopAgentRiskLevel.LOW,
        subgoal: str | None = None,
        success_label: str | None = None,
        alternatives: list[str] | None = None,
    ) -> DesktopAgentStep:
        return DesktopAgentStep(
            step_id=step_id,
            title=title,
            action_type=action_type,
            precondition="desktop runtime available and observation refreshed",
            action=action,
            verification=verification,
            subgoal=subgoal,
            success_label=success_label,
            fallback="re-observe, verify focus, and retry only with a safe alternative",
            alternatives=alternatives or [],
            payload=payload,
            risk_level=risk,
            max_retries=1,
        )

    @staticmethod
    def _is_browser_search(lowered: str) -> bool:
        return "abre " in lowered and "busca " in lowered and any(browser in lowered for browser in ("chrome", "opera", "edge"))

    @staticmethod
    def _is_write_to_active_window(lowered: str) -> bool:
        return "ventana activa" in lowered and ("escribe" in lowered or "escribir" in lowered)

    @staticmethod
    def _is_open_and_type(lowered: str) -> bool:
        return lowered.startswith("abre ") and ("escribe esto" in lowered or "escribir" in lowered)

    @staticmethod
    def _is_file_search_and_open(lowered: str) -> bool:
        return "archivo" in lowered and ("abre" in lowered or "abr" in lowered) and "busca" in lowered

    @staticmethod
    def _is_word_continue(lowered: str) -> bool:
        return "word" in lowered and ("contin" in lowered or "anade" in lowered or "añade" in lowered or "libro" in lowered or "documento" in lowered)

    @staticmethod
    def _extract_browser_search(goal: str) -> tuple[str, str]:
        match = re.search(r"abre\s+(.+?)\s+y\s+busca\s+(.+)$", goal, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return "Chrome", goal

    @staticmethod
    def _extract_application(goal: str) -> str:
        match = re.search(r"abre\s+(.+?)(?:\s+y|$)", goal, flags=re.IGNORECASE)
        return (match.group(1).strip() if match else "").strip("\"' ") or "Word"

    @staticmethod
    def _extract_literal_tail(goal: str) -> str:
        match = re.search(r"(?:escribe esto|anade este texto|añade este texto)\s*:?\s*(.+)$", goal, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_file_query(goal: str) -> str:
        match = re.search(r"busca(?:\s+este)?\s+archivo\s*(.+?)(?:\s+y\s+abrelo|\s+y\s+ábrelo|$)", goal, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" :\"'")
        return goal

    @staticmethod
    def _infer_process_hint(application: str) -> str | None:
        mapping = {
            "chrome": "chrome",
            "opera": "launcher",
            "opera gx": "launcher",
            "word": "winword",
            "vscode": "code",
            "visual studio code": "code",
            "notepad": "notepad",
        }
        lowered = application.casefold()
        for key, value in mapping.items():
            if key in lowered:
                return value
        return lowered or None
