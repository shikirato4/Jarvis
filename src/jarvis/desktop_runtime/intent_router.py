from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from jarvis.core.errors import JarvisError
from jarvis.routing.models import TaskRequest
from jarvis.services import (
    summarize_ops_status,
    summarize_research_task,
    summarize_science_result,
    summarize_security_result,
    summarize_system_operation,
    summarize_system_search,
    summarize_writing_receipt,
)
from jarvis.system_runtime.windows_apps import catalog_descriptor_for_query
from jarvis.web_search import should_use_web_search


@dataclass(frozen=True)
class DesktopIntentDecision:
    category: str
    prompt: str
    target: str | None = None
    mission_id: str | None = None
    target_window: str | None = None
    literal_text: str | None = None
    action: str | None = None
    ui_label: str | None = None
    ui_kind: str | None = None
    text_value: str | None = None
    launch_target: str | None = None
    search_query: str | None = None
    approved: bool = False


class DesktopIntentRouter:
    def __init__(self, bridge, action_executor) -> None:
        self._bridge = bridge
        self._actions = action_executor

    def classify(self, prompt: str) -> DesktopIntentDecision:
        normalized = prompt.strip()
        folded = self._fold(normalized)
        if not normalized:
            return DesktopIntentDecision(category="empty", prompt=normalized)
        literal_text = self._extract_literal_voice_text(normalized, folded)
        if literal_text is not None:
            return DesktopIntentDecision(category="voice_speak_literal", prompt=normalized, literal_text=literal_text)
        if "estado" in folded and "sistema" in folded:
            return DesktopIntentDecision(category="ops_status", prompt=normalized)
        if "diagn" in folded:
            return DesktopIntentDecision(category="diagnostics", prompt=normalized)
        if self._is_writing_inspection(folded):
            return DesktopIntentDecision(
                category="writing_inspect",
                prompt=normalized,
                target_window=self._detect_target_window(normalized, folded),
            )
        if self._is_writing_continuation(folded):
            return DesktopIntentDecision(
                category="writing_continue",
                prompt=normalized,
                target_window=self._detect_target_window(normalized, folded),
            )
        if self._is_self_improvement_request(folded):
            return DesktopIntentDecision(category="self_improvement", prompt=normalized)
        if self._is_window_context_request(folded):
            return DesktopIntentDecision(category="window_context", prompt=normalized)
        if self._is_ui_window_question(folded):
            return DesktopIntentDecision(category="ui_window", prompt=normalized)
        if self._is_screen_read_request(folded):
            return DesktopIntentDecision(category="screen_read", prompt=normalized)
        mission_control = self._extract_mission_control(normalized, folded)
        if mission_control is not None:
            return mission_control
        if self._is_desktop_agent_goal(folded):
            return DesktopIntentDecision(category="desktop_agent", prompt=normalized)
        open_and_search = self._extract_open_and_search(normalized, folded)
        if open_and_search is not None:
            return DesktopIntentDecision(
                category="ui_navigate",
                prompt=normalized,
                action="browser_search",
                launch_target=open_and_search["application"],
                target_window=open_and_search["application"],
                search_query=open_and_search["query"],
            )
        open_and_type = self._extract_open_and_type(normalized, folded)
        if open_and_type is not None:
            return DesktopIntentDecision(
                category="ui_type",
                prompt=normalized,
                launch_target=open_and_type["application"],
                target_window=open_and_type["application"],
                text_value=open_and_type["text"],
            )
        click_target = self._extract_click_target(normalized, folded)
        if click_target is not None:
            return DesktopIntentDecision(
                category="ui_click",
                prompt=normalized,
                ui_label=click_target["label"],
                ui_kind=click_target.get("kind"),
                action=click_target["action"],
                target_window=self._detect_target_window(normalized, folded),
            )
        if self._is_close_window_request(folded):
            return DesktopIntentDecision(
                category="ui_navigate",
                prompt=normalized,
                action="close_window",
                target_window=self._detect_target_window(normalized, folded),
            )
        focus_target = self._extract_focus_window_target(normalized, folded)
        if focus_target is not None:
            return DesktopIntentDecision(
                category="ui_navigate",
                prompt=normalized,
                action="focus_window",
                target_window=focus_target,
            )
        type_request = self._extract_type_request(normalized, folded)
        if type_request is not None:
            return DesktopIntentDecision(
                category="ui_type",
                prompt=normalized,
                text_value=type_request["text"],
                target_window=type_request.get("target_window"),
            )
        if self._is_system_open_request(folded):
            return DesktopIntentDecision(category="system_open", prompt=normalized, target=self._strip_command(normalized, folded))
        if self._is_system_search_request(folded):
            return DesktopIntentDecision(category="system_search", prompt=normalized, target=self._strip_command(normalized, folded))
        if self._is_explicit_research_request(folded):
            return DesktopIntentDecision(category="research", prompt=normalized)
        if self._is_security_query(folded):
            return DesktopIntentDecision(category="security", prompt=normalized)
        if self._is_science_query(folded):
            return DesktopIntentDecision(category="science", prompt=normalized)
        return DesktopIntentDecision(category="chat", prompt=normalized)

    def execute(self, decision: DesktopIntentDecision) -> tuple[dict, str, str]:
        runtime = self._bridge.runtime
        if decision.category in {"ui_click", "ui_type", "ui_navigate"}:
            result = runtime.desktop_agent_run(
                {
                    "goal": decision.prompt,
                    "metadata": {"source": "desktop_chat"},
                    "source_surface": "desktop_chat",
                }
            ).model_dump(mode="json")
            return result, self._summarize_desktop_agent(result), "desktop_agent_runtime.run"
        if decision.category == "empty":
            return {}, "Estoy listo.", "desktop.empty"
        if decision.category == "voice_speak_literal":
            literal_text = (decision.literal_text or "").strip()
            return (
                {
                    "ok": True,
                    "category": "voice_speak_literal",
                    "literal_text": literal_text,
                    "message": "Reproduciendo mensaje solicitado.",
                    "speak_only": True,
                },
                "Reproduciendo mensaje solicitado.",
                "voice_runtime.literal",
            )
        if decision.category == "ops_status":
            result = runtime.ops_status()
            return result, summarize_ops_status(result), "ops_status"
        if decision.category == "diagnostics":
            result = self._actions.execute("ops.diagnostics")
            return result, "Diagnostics listo. Se actualizaron los paneles de control.", "diagnostics"
        if decision.category == "research":
            result = self._actions.execute("research.run", payload={"query": decision.prompt})
            return result, summarize_research_task(result), "research_runtime"
        if decision.category == "science":
            result = self._execute_science(decision.prompt)
            return result, summarize_science_result(result), "science_runtime"
        if decision.category == "security":
            result = self._execute_security(decision.prompt)
            return result, self._summarize_security_result(result), "security_runtime"
        if decision.category == "system_open":
            result = self._execute_system_open(decision)
            return result, summarize_system_operation(result), "system_runtime.open"
        if decision.category == "system_search":
            result = runtime.system_search(
                {"resource": {"query": decision.target or decision.prompt}, "metadata": {"source": "desktop_chat"}}
            ).model_dump(mode="json")
            return result, summarize_system_search(result), "system_runtime.search"
        if decision.category == "ui_window":
            result = runtime.ui_active_window().model_dump(mode="json")
            return result, self._summarize_active_window(result), "ui_automation.active_window"
        if decision.category == "window_context":
            try:
                receipt = runtime.vision_describe_active_window()
                result = receipt.model_dump(mode="json", exclude={"capture_result": {"image_bytes"}})
                result.setdefault("status", "completed")
                result.setdefault("plan", {"strategy": "grounded_screen_read"})
            except JarvisError as exc:
                if exc.component != "vision_runtime" or exc.code != "capture_failed":
                    raise
                active_result = runtime.ui_active_window().model_dump(mode="json")
                active_window = active_result.get("active_window")
                result = {
                    "ok": True,
                    "status": "completed",
                    "plan": {"strategy": "grounded_screen_read"},
                    "operation_name": "vision.describe_active_window",
                    "degraded": True,
                    "degradation_reason": exc.message,
                    "fallback_operation": "ui_automation.active_window",
                    "active_window": active_window,
                    "awareness_result": {
                        "window": active_window,
                        "summary": "",
                    },
                }
            except Exception as exc:  # noqa: BLE001
                result = {
                    "ok": False,
                    "status": "completed",
                    "plan": {"strategy": "grounded_screen_read"},
                    "operation_name": "vision.describe_active_window",
                    "error": {"message": str(exc), "type": type(exc).__name__},
                }
            return result, self._summarize_window_context(result), "vision_runtime.describe_active_window"
        if decision.category == "screen_read":
            result = runtime.vision_extract_text(
                {"capture": {"target_type": "active_window"}, "metadata": {"source": "desktop_chat"}}
            ).model_dump(mode="json")
            return result, self._summarize_screen_text(result), "vision_runtime.extract_text"
        if decision.category == "desktop_agent":
            result = runtime.desktop_agent_run(
                {
                    "goal": decision.prompt,
                    "metadata": {"source": "desktop_chat"},
                    "source_surface": "desktop_chat",
                }
            ).model_dump(mode="json")
            return result, self._summarize_desktop_agent(result), "desktop_agent_runtime.run"
        if decision.category == "desktop_agent_pause":
            result = runtime.desktop_agent_pause(decision.mission_id or self._latest_desktop_agent_mission_id()).model_dump(mode="json")
            return result, self._summarize_desktop_agent_control("pausada", result), "desktop_agent_runtime.pause"
        if decision.category == "desktop_agent_resume":
            result = runtime.desktop_agent_resume(decision.mission_id or self._latest_desktop_agent_mission_id()).model_dump(mode="json")
            return result, self._summarize_desktop_agent_control("reanudada", result), "desktop_agent_runtime.resume"
        if decision.category == "desktop_agent_confirm":
            result = runtime.desktop_agent_confirm(decision.mission_id or self._latest_desktop_agent_mission_id()).model_dump(mode="json")
            return result, self._summarize_desktop_agent(result), "desktop_agent_runtime.confirm"
        if decision.category == "desktop_agent_abort":
            result = runtime.desktop_agent_abort(decision.mission_id or self._latest_desktop_agent_mission_id()).model_dump(mode="json")
            return result, self._summarize_desktop_agent_control("abortada", result), "desktop_agent_runtime.abort"
        if decision.category == "desktop_agent_status":
            result = runtime.desktop_agent_get(decision.mission_id or self._latest_desktop_agent_mission_id()).model_dump(mode="json")
            return result, self._summarize_desktop_agent_status(result), "desktop_agent_runtime.status"
        if decision.category == "desktop_agent_list":
            missions = [mission.model_dump(mode="json") for mission in runtime.desktop_agent_list()]
            return {"missions": missions}, self._summarize_desktop_agent_list(missions), "desktop_agent_runtime.list"
        if decision.category == "ui_click":
            self._ensure_operator_mode()
            result = runtime.ui_click_target(
                {
                    "label": decision.ui_label or "",
                    "kind": decision.ui_kind,
                    "double": decision.action == "double_click",
                    "target_window": decision.target_window,
                    "approved": decision.approved,
                }
            ).model_dump(mode="json")
            return result, self._summarize_ui_click(decision, result), "ui_automation.click_target"
        if decision.category == "ui_type":
            result = self._execute_ui_type(decision)
            return result, self._summarize_ui_type(decision, result), "ui_automation.write_text"
        if decision.category == "ui_navigate":
            result = self._execute_ui_navigation(decision)
            return result, self._summarize_ui_navigation(decision, result), "ui_automation.navigate"
        if decision.category == "writing_inspect":
            result = runtime.writing_analyze(
                {
                    "prompt": decision.prompt,
                    "target_window": decision.target_window,
                    "write_directly": False,
                    "metadata": {"source": "desktop_chat", "action": "inspect_context"},
                }
            ).model_dump(mode="json")
            return result, self._summarize_writing_analysis(result), "writing_runtime.analyze"
        if decision.category == "writing_continue":
            result = runtime.writing_continue(
                {
                    "prompt": decision.prompt,
                    "target_window": decision.target_window or self._active_window_title(),
                    "write_directly": True,
                    "metadata": {"source": "desktop_chat", "action": "continue", "approved": decision.approved},
                }
            ).model_dump(mode="json")
            return result, summarize_writing_receipt(result, active_title=result.get("window_title")), "writing_runtime.continue"
        if decision.category == "self_improvement":
            result = runtime.self_improvement_run({"prompt": decision.prompt, "metadata": {"source": "desktop_chat"}}).model_dump(mode="json")
            return result, self._summarize_self_improvement(result), "self_improvement_runtime.run"
        raise ValueError(f"unsupported desktop intent category: {decision.category}")

    def _execute_ui_type(self, decision: DesktopIntentDecision) -> dict:
        runtime = self._bridge.runtime
        if decision.launch_target:
            self._execute_system_open(
                DesktopIntentDecision(category="system_open", prompt=decision.prompt, target=decision.launch_target, approved=decision.approved)
            )
            time.sleep(0.1)
        target_window = decision.target_window or decision.launch_target or self._active_window_title()
        receipt = runtime.ui_write_text(
            {
                "text": decision.text_value or "",
                "mode": "copilot",
                "target_window": target_window,
                "typing_interval_ms": 0,
                "pause_between_blocks_ms": 0,
                "approved": decision.approved or self._is_trusted_application_target(target_window or ""),
            },
            correlation_id="desktop-ui-type",
        ).model_dump(mode="json")
        if decision.launch_target:
            receipt["launch_target"] = decision.launch_target
        return receipt

    def _execute_ui_navigation(self, decision: DesktopIntentDecision) -> dict:
        runtime = self._bridge.runtime
        if decision.action == "focus_window":
            return runtime.ui_focus_window({"target": decision.target_window or "", "approved": decision.approved}).model_dump(mode="json")
        if decision.action == "close_window":
            self._ensure_operator_mode()
            return runtime.ui_close_window({"target": decision.target_window, "approved": decision.approved}).model_dump(mode="json")
        if decision.action == "browser_search":
            self._ensure_operator_mode()
            launch = self._execute_system_open(
                DesktopIntentDecision(category="system_open", prompt=decision.prompt, target=decision.launch_target, approved=decision.approved)
            )
            time.sleep(0.1)
            focus = runtime.ui_focus_window({"target": decision.target_window or decision.launch_target or "", "approved": True}).model_dump(mode="json")
            shortcut = runtime.ui_hotkey({"keys": ("ctrl", "l"), "approved": True}).model_dump(mode="json")
            typed = runtime.ui_write_text(
                {
                    "text": decision.search_query or "",
                    "mode": "copilot",
                    "target_window": decision.target_window or decision.launch_target,
                    "typing_interval_ms": 0,
                    "pause_between_blocks_ms": 0,
                    "approved": True,
                },
                correlation_id="desktop-ui-search",
            ).model_dump(mode="json")
            submit = runtime.ui_hotkey({"keys": ("enter",), "approved": True}).model_dump(mode="json")
            return {
                "success": True,
                "action": "browser_search",
                "launch": launch,
                "focus": focus,
                "address_bar": shortcut,
                "typed": typed,
                "submit": submit,
                "query": decision.search_query,
                "target_window": decision.target_window or decision.launch_target,
            }
        raise ValueError(f"unsupported ui navigation action: {decision.action}")

    def _execute_system_open(self, decision: DesktopIntentDecision) -> dict:
        runtime = self._bridge.runtime
        target = self._canonical_application_target((decision.target or "").strip())
        approved = decision.approved or self._is_trusted_application_target(target)
        metadata = {"source": "desktop_chat", "approved": approved}
        if self._looks_like_application_target(target):
            return runtime.system_open_application(target, metadata=metadata).model_dump(mode="json")
        return runtime.system_open({"query": target or decision.prompt, "metadata": metadata}).model_dump(mode="json")

    def _execute_security(self, prompt: str) -> dict:
        runtime = self._bridge.runtime
        folded = self._fold(prompt)
        if "contrasena" in folded or "password" in folded:
            secret = self._extract_password_candidate(prompt)
            if secret:
                return runtime.security_check_password({"password": secret}).model_dump(mode="json")
        return self._bridge.route_task(
            TaskRequest(raw_input=prompt, intent="security", metadata={"source": "desktop_chat", "surface": "desktop"})
        ).model_dump(mode="json")

    def _execute_science(self, prompt: str) -> dict:
        runtime = self._bridge.runtime
        folded = self._fold(prompt)
        if "simula" in folded:
            return runtime.science_simulate({"query": prompt}).model_dump(mode="json")
        return runtime.science_solve({"query": prompt}).model_dump(mode="json")

    @staticmethod
    def _is_desktop_agent_goal(folded: str) -> bool:
        return (
            ("abre " in folded and " y " in folded)
            or "ventana activa" in folded
            or ("archivo" in folded and "abre" in folded and "busca" in folded)
            or ("archivo" in folded and any(token in folded for token in ("abre", "abrir")) and "busca" not in folded)
            or ("archivo" in folded and any(token in folded for token in ("crea", "crear", "copia", "copiar", "mueve", "mover", "renombra", "renombrar")))
            or ("carpeta" in folded and any(token in folded for token in ("abre", "abrir", "crea", "crear", "mueve", "mover")))
            or "explorador" in folded
            or "explorer" in folded
            or ("paso a paso" in folded and "abre " in folded)
            or (any(token in folded for token in ("haz click", "haz clic", "click en", "clic en")) and any(token in folded for token in ("boton", "button", "campo", "input", "enviar", "send")))
            or any(token in folded for token in ("llena este formulario", "rellena este formulario", "completa este formulario", "fill this form"))
        )

    def _ensure_operator_mode(self) -> None:
        runtime = self._bridge.runtime
        try:
            runtime.switch_mode("operator", reason="desktop operational intent", sticky=False)
        except Exception:  # noqa: BLE001
            return

    def _summarize_security_result(self, result: dict) -> str:
        orchestration = result.get("orchestration") or {}
        receipts = orchestration.get("receipts") or []
        if receipts:
            payload = receipts[-1].get("data") or {}
            return summarize_security_result(payload)
        return summarize_security_result(result)

    @staticmethod
    def _summarize_active_window(result: dict) -> str:
        active_window = (result.get("active_window") or {}).get("title")
        if active_window:
            return f"La ventana activa es {active_window}."
        return "No detecte una ventana activa en este momento."

    @staticmethod
    def _summarize_screen_text(result: dict) -> str:
        text = ((result.get("ocr_result") or {}).get("text") or "").strip()
        if text:
            preview = text.replace("\n", " ")
            return f"He leido texto visible: {preview[:220]}"
        return "No detecte texto util en la ventana activa."

    @staticmethod
    def _summarize_window_context(result: dict) -> str:
        awareness = result.get("awareness_result") or {}
        window = awareness.get("window") or {}
        title = window.get("title") or "la ventana activa"
        if result.get("degraded"):
            reason = str(result.get("degradation_reason") or "").casefold()
            if "sensitive" in reason or "sensible" in reason or "protected" in reason or "proteg" in reason:
                return "No pude capturar esa ventana porque parece sensible o protegida."
            if window.get("title"):
                return f"Veo {title}. No pude obtener una captura visual completa."
            return "Veo la pantalla, pero no pude obtener una captura visual completa ni detectar la ventana activa."
        if result.get("ok") is False:
            return "Veo la pantalla, pero no pude obtener una captura visual en este momento."
        summary = str(awareness.get("summary") or "").strip()
        elements = awareness.get("elements") or []
        if summary:
            return f"Veo {title}. {summary}"
        if elements:
            labels = [item.get("label") or item.get("text") for item in elements if item.get("label") or item.get("text")]
            preview = ", ".join(labels[:4])
            return f"Veo {title}. Elementos detectados: {preview}."
        return f"Veo {title}, pero no detecte suficiente contexto visual."

    @staticmethod
    def _summarize_ui_click(decision: DesktopIntentDecision, result: dict) -> str:
        label = (result.get("data") or {}).get("matched_label") or decision.ui_label or "el objetivo solicitado"
        if result.get("confirmation_required"):
            return "Esta aplicacion requiere permiso. Deseas continuar?"
        action = "doble click" if decision.action == "double_click" else "click"
        return f"Entendi {action} en {label}. Accion ejecutada."

    @staticmethod
    def _summarize_ui_type(decision: DesktopIntentDecision, result: dict) -> str:
        target_window = (result.get("active_window") or {}).get("title") or decision.target_window or decision.launch_target or "la ventana activa"
        if result.get("confirmation_required"):
            return "Esta aplicacion requiere permiso. Deseas continuar?"
        if decision.launch_target:
            return f"Abriendo {decision.launch_target}. Escribiendo en {target_window}. Listo."
        return f"Escribiendo en {target_window}. Listo."

    @staticmethod
    def _summarize_ui_navigation(decision: DesktopIntentDecision, result: dict) -> str:
        if result.get("confirmation_required"):
            return "Esta aplicacion requiere permiso. Deseas continuar?"
        if decision.action == "focus_window":
            return f"Cambiando a {decision.target_window}. Listo."
        if decision.action == "close_window":
            return "Cerrando la ventana actual. Listo."
        if decision.action == "browser_search":
            return f"Abriendo {decision.launch_target}. Navegando a {decision.search_query}. Listo."
        return "Navegacion de interfaz completada."

    @staticmethod
    def _summarize_writing_analysis(result: dict) -> str:
        context = result.get("context") or {}
        style = result.get("style_profile") or {}
        integration = ((context.get("metadata") or {}).get("integration") or "").strip()
        window_title = context.get("window_title") or "la ventana activa"
        application_name = context.get("application_name") or "editor"
        snippet = (context.get("combined_context") or context.get("visible_text") or context.get("recent_text") or "").strip()
        if integration == "word_com" and snippet:
            preview = snippet.replace("\n", " ")
            return f"He leido el documento activo de Word. Vista previa: {preview[:220]}"
        if snippet:
            preview = snippet.replace("\n", " ")
            return (
                f"He detectado {application_name} como contexto de escritura en {window_title}. "
                f"Tengo suficiente contexto para continuar. Vista previa: {preview[:220]}"
            )
        text_type = style.get("text_type") or "texto"
        return f"He detectado {application_name} en {window_title}. El contexto parece ser {text_type}."

    @staticmethod
    def _summarize_self_improvement(result: dict) -> str:
        issues = result.get("issues") or []
        proposal = result.get("proposal") or {}
        policy = result.get("policy") or {}
        baseline = result.get("baseline_tests") or {}
        candidate = result.get("candidate_tests") or {}
        comparison = result.get("comparison") or {}
        validation = result.get("data", {}).get("validation_summary") or {}
        lines: list[str] = []
        if issues:
            first = issues[0]
            lines.append(f"Detecte un problema en {first.get('file_path')}.")
        if proposal.get("summary"):
            lines.append(f"Propuse este cambio: {proposal['summary']}")
        if policy.get("risk_level"):
            lines.append(f"Riesgo estimado: {policy.get('risk_level')}.")
        if policy.get("warnings"):
            lines.append(f"Alertas: {' | '.join(policy.get('warnings')[:3])}")
        if proposal.get("diff"):
            diff_lines = str(proposal["diff"]).splitlines()
            preview = "\n".join(diff_lines[:12])
            lines.append(f"Diff generado:\n{preview}")
        if baseline.get("summary") or candidate.get("summary"):
            lines.append(
                "Ejecute tests: "
                f"baseline={baseline.get('summary') or 'sin resumen'}; "
                f"candidato={candidate.get('summary') or 'sin resumen'}"
            )
        if validation:
            lines.append(
                "Validaciones: "
                f"sintaxis={validation.get('syntax_ok')}; "
                f"compileall={validation.get('compileall_ok')}; "
                f"imports={validation.get('imports_ok')}; "
                f"suite_verde={validation.get('candidate_green')}"
            )
        if comparison.get("notes"):
            lines.append(f"Notas de validacion: {' | '.join(comparison.get('notes')[:3])}")
        if result.get("approval_decision"):
            lines.append(f"Recomendacion final: {result.get('approval_decision')}.")
        if result.get("message"):
            lines.append(str(result["message"]))
        return "\n".join(lines) if lines else "Revision de automejora completada."

    @staticmethod
    def _summarize_desktop_agent(result: dict) -> str:
        summary = str(result.get("summary") or "").strip()
        if summary:
            return summary
        if result.get("success"):
            return "La mision de escritorio se completo con verificacion por pasos."
        return "La mision de escritorio se detuvo por verificacion o politica."

    @staticmethod
    def _summarize_desktop_agent_control(action: str, result: dict) -> str:
        return f"Mision {action}: {result.get('mission_id')}."

    @staticmethod
    def _summarize_desktop_agent_status(result: dict) -> str:
        progress = result.get("progress") or {}
        return (
            f"Mision {result.get('mission_id')} en estado {result.get('status')}. "
            f"Subtarea actual: {result.get('current_subtask_label') or 'sin subtarea activa'}. "
            f"Progreso: {progress.get('completed_steps', 0)}/{progress.get('total_steps', 0)} pasos."
        )

    @staticmethod
    def _summarize_desktop_agent_list(missions: list[dict]) -> str:
        if not missions:
            return "No hay misiones persistentes del desktop agent."
        latest = missions[-1]
        return (
            f"Hay {len(missions)} misiones persistentes. "
            f"La mas reciente esta en estado {latest.get('status')} y objetivo '{latest.get('goal')}'."
        )

    def _active_window_title(self) -> str | None:
        try:
            receipt = self._bridge.runtime.ui_active_window()
            return receipt.active_window.title if receipt.active_window else None
        except Exception:  # noqa: BLE001
            return None

    def _latest_desktop_agent_mission_id(self) -> str:
        missions = self._bridge.runtime.desktop_agent_list()
        if not missions:
            raise ValueError("no hay misiones del desktop agent para controlar")
        return missions[-1].mission_id

    @staticmethod
    def _fold(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        stripped = "".join(char for char in normalized if not unicodedata.combining(char))
        return stripped.casefold()

    @staticmethod
    def _is_explicit_research_request(folded: str) -> bool:
        return any(keyword in folded for keyword in ("investiga", "investigar", "research", "fuentes", "evidencia"))

    @staticmethod
    def _is_science_query(folded: str) -> bool:
        return any(keyword in folded for keyword in ("calcula", "simula", "estima", "resuelve", "deriv", "integr", "ecuacion"))

    @staticmethod
    def _is_security_query(folded: str) -> bool:
        return any(keyword in folded for keyword in ("seguridad", "vulnerabilidad", "password", "contrasena", "cve", "exploit", "riesgo"))

    @staticmethod
    def _is_writing_continuation(folded: str) -> bool:
        continue_terms = ("continua", "sigue escribiendo", "sigue donde", "continua mi", "continua este")
        document_terms = ("libro", "historia", "documento", "texto", "word", "novela", "cuento")
        return any(term in folded for term in continue_terms) or ("sigue" in folded and any(term in folded for term in document_terms))

    @staticmethod
    def _is_writing_inspection(folded: str) -> bool:
        inspect_terms = ("ves", "puedes ver", "lee", "leer", "que tengo abierta", "que tengo abierto", "historia abierta", "texto abierto")
        document_terms = ("historia", "word", "documento", "libro", "texto")
        return any(term in folded for term in inspect_terms) and any(term in folded for term in document_terms)

    @staticmethod
    def _is_self_improvement_request(folded: str) -> bool:
        return any(
            phrase in folded
            for phrase in (
                "mejora el sistema",
                "encuentra bugs",
                "optimiza este modulo",
                "revisa el codigo",
                "auto mejora",
                "automejora",
            )
        )

    @staticmethod
    def _is_ui_window_question(folded: str) -> bool:
        if DesktopIntentRouter._looks_like_metaphorical_vision_request(folded):
            return False
        exact_phrases = (
            "que ventana esta activa",
            "cual es la ventana activa",
            "que app tengo abierta",
            "que aplicacion esta abierta",
        )
        if any(phrase in folded for phrase in exact_phrases):
            return True
        return "que tengo abierto" in folded and not DesktopIntentRouter._looks_like_visual_context_request(folded)

    @staticmethod
    def _is_window_context_request(folded: str) -> bool:
        if DesktopIntentRouter._looks_like_metaphorical_vision_request(folded):
            return False
        if DesktopIntentRouter._looks_like_screen_read_request(folded):
            return False
        if any(
            phrase in folded
            for phrase in (
                "que hay en mi pantalla",
                "que hay en pantalla",
                "que dice esta ventana",
                "describe esta ventana",
                "describe mi pantalla",
                "que ves en la ventana actual",
                "que ves en la ventana activa",
                "que ves en mi escritorio",
                "que hay en mi escritorio",
                "que hay en el escritorio",
                "analiza lo que tengo abierto",
                "analiza la pantalla",
                "analiza esta pantalla",
                "analiza esta ventana",
                "puedes ver lo que hay en mi escritorio",
                "puedes ver lo que hay en mi pantalla",
            )
        ):
            return True
        if not DesktopIntentRouter._looks_like_visual_context_request(folded):
            return False
        return DesktopIntentRouter._contains_any(
            folded,
            (
                "ves",
                "ver",
                "mira",
                "describe",
                "analiza",
                "que hay",
                "que ves",
                "que tengo abierto",
                "que hay abierto",
                "abierto",
                "visible",
            ),
        )

    @staticmethod
    def _is_screen_read_request(folded: str) -> bool:
        return DesktopIntentRouter._looks_like_screen_read_request(folded)

    @staticmethod
    def _is_system_open_request(folded: str) -> bool:
        return folded.startswith(("abre ", "open "))

    @staticmethod
    def _is_system_search_request(folded: str) -> bool:
        if any(marker in folded for marker in (" en el sistema", " sistema", "archivo", ".txt", ".pdf", ".docx", ".xlsx", ".py", ".json")):
            return folded.startswith(("busca ", "search ", "find "))
        if should_use_web_search(folded, mode="auto"):
            return False
        return folded.startswith(("busca ", "search ", "find "))

    @staticmethod
    def _is_close_window_request(folded: str) -> bool:
        return any(phrase in folded for phrase in ("cierra esta ventana", "cierra la ventana", "close this window", "close window"))

    def _extract_mission_control(self, prompt: str, folded: str) -> DesktopIntentDecision | None:
        mission_id = self._extract_mission_id(prompt)
        if folded in {"detente", "para", "stop", "no hagas nada"} or any(
            phrase in folded for phrase in ("cancela el agente", "aborta el agente", "stop agent")
        ):
            return DesktopIntentDecision(category="desktop_agent_abort", prompt=prompt, mission_id=mission_id)
        if folded in {"continua", "continuar", "confirmo"} or any(
            phrase in folded for phrase in ("confirm action", "confirma la accion", "confirmar accion", "continua agente")
        ):
            return DesktopIntentDecision(category="desktop_agent_confirm", prompt=prompt, mission_id=mission_id)
        if "lista" in folded and "misiones" in folded:
            return DesktopIntentDecision(category="desktop_agent_list", prompt=prompt)
        if "estado" in folded and "mision" in folded:
            return DesktopIntentDecision(category="desktop_agent_status", prompt=prompt, mission_id=mission_id)
        if "pausa" in folded and "mision" in folded:
            return DesktopIntentDecision(category="desktop_agent_pause", prompt=prompt, mission_id=mission_id)
        if ("reanuda" in folded or "resume" in folded) and "mision" in folded:
            return DesktopIntentDecision(category="desktop_agent_resume", prompt=prompt, mission_id=mission_id)
        if ("aborta" in folded or "cancela" in folded) and "mision" in folded:
            return DesktopIntentDecision(category="desktop_agent_abort", prompt=prompt, mission_id=mission_id)
        return None

    @classmethod
    def _extract_literal_voice_text(cls, prompt: str, folded: str) -> str | None:
        prefix_patterns = ("di esto:", "habla esto:", "repite:", "di:", "solo di ")
        for prefix in prefix_patterns:
            if folded.startswith(prefix):
                literal = prompt[len(prefix) :].strip()
                return literal or None
        if folded.startswith("di ") and len(prompt.split()) > 1:
            literal = prompt[3:].strip()
            return literal or None
        return None

    @classmethod
    def _strip_command(cls, prompt: str, folded: str) -> str:
        prefixes = ("abre ", "open ", "busca ", "search ", "find ")
        for prefix in prefixes:
            if folded.startswith(prefix):
                return prompt[len(prefix) :].strip()
        return prompt.strip()

    @classmethod
    def _extract_open_and_search(cls, prompt: str, folded: str) -> dict[str, str] | None:
        if not folded.startswith("abre "):
            return None
        separators = (" y busca ", " y search ", " and search ")
        for separator in separators:
            index = folded.find(separator)
            if index >= 0:
                left = prompt[:index]
                right = prompt[index + len(separator) :]
                application = cls._strip_command(left.strip(), cls._fold(left.strip()))
                query = right.strip(" :")
                if application and query:
                    return {"application": application, "query": query}
        return None

    @classmethod
    def _extract_open_and_type(cls, prompt: str, folded: str) -> dict[str, str] | None:
        if not folded.startswith("abre "):
            return None
        separators = (" y escribe esto:", " y escribe esto ", " y escribe ", " and type ")
        for separator in separators:
            index = folded.find(separator)
            if index >= 0:
                left = prompt[:index]
                right = prompt[index + len(separator) :]
                application = cls._strip_command(left.strip(), cls._fold(left.strip()))
                text = right.strip()
                if application and text:
                    return {"application": application, "text": text}
        return None

    @classmethod
    def _extract_click_target(cls, prompt: str, folded: str) -> dict[str, str] | None:
        if "click" not in folded and "clic" not in folded:
            return None
        action = "double_click" if "doble" in folded or "double" in folded else "click"
        kind = None
        label = prompt
        markers = (
            ("boton de ", "button"),
            ("boton del ", "button"),
            ("boton ", "button"),
            ("button ", "button"),
            ("campo ", "input"),
            ("input ", "input"),
            ("texto ", "text"),
        )
        for marker, detected_kind in markers:
            marker_index = folded.rfind(marker)
            if marker_index >= 0:
                label = prompt[marker_index + len(marker) :]
                kind = detected_kind
                break
        else:
            generic_markers = ("click en ", "clic en ", "click on ", "haz click en ", "haz clic en ")
            for marker in generic_markers:
                marker_index = folded.rfind(marker)
                if marker_index >= 0:
                    label = prompt[marker_index + len(marker) :]
                    break
        cleaned = cls._clean_label(label)
        if not cleaned:
            return None
        return {"label": cleaned, "kind": kind, "action": action}

    @classmethod
    def _extract_type_request(cls, prompt: str, folded: str) -> dict[str, str] | None:
        if not folded.startswith(("escribe ", "type ")):
            return None
        prefixes = ("escribe esto:", "escribe esto ", "escribe ", "type ")
        text = None
        for prefix in prefixes:
            if folded.startswith(prefix):
                text = prompt[len(prefix) :].strip()
                break
        if not text:
            return None
        return {"text": text, "target_window": cls._detect_target_window(prompt, folded)}

    @classmethod
    def _extract_focus_window_target(cls, prompt: str, folded: str) -> str | None:
        markers = ("cambia a ", "ve a ", "enfoca ", "focus ")
        for marker in markers:
            if folded.startswith(marker):
                target = prompt[len(marker) :].strip()
                return target or None
        return None

    @staticmethod
    def _clean_label(label: str) -> str:
        cleaned = re.sub(r"^[\s:]+", "", label.strip())
        cleaned = re.sub(r"^(el|la|los|las|al|del)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+en\s+(word|notepad|chrome|opera|vscode|vs code|documento)$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" .")

    @staticmethod
    def _detect_target_window(prompt: str, folded: str) -> str | None:
        if "word" in folded:
            return "Word"
        if "notepad" in folded or "bloc de notas" in folded:
            return "Notepad"
        if "vscode" in folded or "vs code" in folded or "visual studio code" in folded:
            return "VSCode"
        if "chrome" in folded:
            return "Chrome"
        if "opera" in folded:
            return "Opera"
        if "documento" in folded:
            return "documento"
        return None

    @staticmethod
    def _canonical_application_target(target: str) -> str:
        cleaned = target.strip()
        descriptor = catalog_descriptor_for_query(cleaned)
        if descriptor is not None:
            return descriptor.canonical_id
        lowered = cleaned.casefold()
        for prefix in ("el ", "la ", "los ", "las "):
            if lowered.startswith(prefix):
                descriptor = catalog_descriptor_for_query(cleaned[len(prefix) :].strip())
                if descriptor is not None:
                    return descriptor.canonical_id
        return cleaned

    @staticmethod
    def _is_trusted_application_target(target: str) -> bool:
        lowered = target.casefold().strip()
        return lowered in {"winword", "word", "notepad", "calc", "code", "vscode", "chrome", "opera", "explorer"}

    @staticmethod
    def _looks_like_application_target(target: str) -> bool:
        lowered = target.casefold().strip()
        if not lowered:
            return False
        if any(token in lowered for token in ("\\", "/", ":", ".txt", ".doc", ".docx", ".md", "carpeta", "folder")):
            return False
        return len(lowered.split()) <= 3

    @staticmethod
    def _extract_password_candidate(prompt: str) -> str | None:
        cleaned = prompt.replace("?", " ").replace(",", " ")
        tokens = [token for token in cleaned.split() if token]
        if not tokens:
            return None
        markers = {"contrasena", "password", "clave"}
        for index, token in enumerate(tokens):
            if DesktopIntentRouter._fold(token) in markers and index + 1 < len(tokens):
                return tokens[index + 1]
        return tokens[-1]

    @staticmethod
    def _extract_mission_id(prompt: str) -> str | None:
        match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", prompt, flags=re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _looks_like_metaphorical_vision_request(folded: str) -> bool:
        return DesktopIntentRouter._contains_any(
            folded,
            (
                "ves lo que digo",
                "ves lo que quiero decir",
                "ves por donde voy",
                "ves mi punto",
                "do you see what i mean",
            ),
        )

    @staticmethod
    def _looks_like_visual_context_request(folded: str) -> bool:
        if DesktopIntentRouter._looks_like_metaphorical_vision_request(folded):
            return False
        surface_terms = (
            "pantalla",
            "screen",
            "escritorio",
            "desktop",
            "ventana",
            "window",
            "ventana actual",
            "ventana activa",
            "lo que tengo abierto",
            "que tengo abierto",
            "que hay abierto",
            "contenido visible",
            "visible",
        )
        observation_terms = (
            "ves",
            "ver",
            "puedes ver",
            "que ves",
            "que hay",
            "analiza",
            "analizar",
            "describe",
            "mostrar",
            "muestra",
            "inspecciona",
            "abierto",
            "leer",
            "lee",
            "dice",
        )
        return DesktopIntentRouter._contains_any(folded, surface_terms) and DesktopIntentRouter._contains_any(folded, observation_terms)

    @staticmethod
    def _looks_like_screen_read_request(folded: str) -> bool:
        if DesktopIntentRouter._looks_like_metaphorical_vision_request(folded):
            return False
        direct_phrases = (
            "lee esto de la pantalla",
            "lee la pantalla",
            "lee esta ventana",
            "lee esta pantalla",
            "que dice la pantalla",
            "que dice esta pantalla",
            "lee lo que hay en pantalla",
            "lee lo que hay en pantalla",
            "lee lo que hay en mi pantalla",
        )
        if any(phrase in folded for phrase in direct_phrases):
            return True
        read_terms = ("lee", "leer", "que dice", "texto", "ocr")
        surface_terms = ("pantalla", "screen", "ventana", "window", "escritorio", "desktop")
        return DesktopIntentRouter._contains_any(folded, surface_terms) and DesktopIntentRouter._contains_any(folded, read_terms)
