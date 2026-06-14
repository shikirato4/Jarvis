from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from uuid import uuid4

from jarvis.config import Settings
from jarvis.chat_optimization import select_chat_generation_profile, streaming_timeout_seconds
from jarvis.core.errors import JarvisError
from jarvis.identity import jarvis_identity_prompt, sanitize_assistant_identity
from jarvis.models_runtime.base import ModelRequest
from jarvis.ollama_diagnostics import classify_local_model_error, local_model_failure_message
from jarvis.services import summarize_error, summarize_model_response
from jarvis.voice_runtime.spoken import prepare_spoken_response
from jarvis.web_search import build_grounded_web_prompt, build_web_search_provider, should_use_web_search

from .base import DesktopChatMessage, DesktopChatResponse
from .intent_router import DesktopIntentDecision, DesktopIntentRouter


@dataclass
class PendingDesktopAction:
    decision: DesktopIntentDecision
    prompt: str
    runtime_name: str | None = None
    target_window: str | None = None


class DesktopChatEngine:
    def __init__(self, bridge, panel_composer, action_executor, settings: Settings) -> None:
        self._bridge = bridge
        self._panels = panel_composer
        self._actions = action_executor
        self._settings = settings
        self._intent_router = DesktopIntentRouter(bridge, action_executor)
        self._logger = logging.getLogger("jarvis.desktop.chat")
        self.pending_action: PendingDesktopAction | None = None
        self.awaiting_confirmation = False

    def handle(self, text: str) -> DesktopChatResponse:
        normalized = text.strip()
        result: dict = {}
        response_text = ""
        web_result_for_error = None
        spoken_content_override: str | None = None
        spoken_mode = "prepared"
        if self.awaiting_confirmation and self.pending_action is not None:
            return self._handle_confirmation_reply(normalized)
        decision = self._intent_router.classify(normalized)
        self._logger.info("desktop_intent_detected", extra={"intent": decision.category, "prompt": normalized})
        try:
            if decision.category == "chat":
                if self._is_context_query(lowered := normalized.casefold()):
                    web_status = build_web_search_provider().status()
                    result = {"status": "ok", "context": "jarvis_context", "web_search": web_status.model_dump(mode="json")}
                    response_text = self._context_summary(web_status)
                else:
                    request, web_result = self._build_model_request(normalized)
                    web_result_for_error = web_result
                    route = f"models_runtime:{request.logical_model}"
                    if web_result is not None:
                        route = f"web_search:{web_result.provider}+{route}"
                    self._log_route(route, normalized)
                    model_response = self._bridge.runtime.infer_model(request)
                    result = model_response.model_dump(mode="json")
                    if web_result is not None:
                        result["web_search"] = web_result.model_dump(mode="json")
                    response_text = sanitize_assistant_identity(summarize_model_response(result))
            else:
                result, response_text, route_name = self._intent_router.execute(decision)
                self._log_route(route_name, normalized)
                self._log_execution(route_name, normalized, result)
                if decision.category == "voice_speak_literal":
                    spoken_content_override = decision.literal_text or ""
                    spoken_mode = "literal"
                if self._requires_confirmation(result):
                    self.pending_action = PendingDesktopAction(
                        decision=decision,
                        prompt=normalized,
                        runtime_name=route_name,
                        target_window=self._target_window_from_result(result),
                    )
                    self.awaiting_confirmation = True
                    response_text = self._confirmation_prompt(result)
        except JarvisError as exc:
            result = {"error": exc.to_dict(), "ok": False}
            response_text = sanitize_assistant_identity(summarize_error(normalized, str(exc)))
        except Exception as exc:  # noqa: BLE001
            result = {"error": {"message": str(exc), "type": type(exc).__name__}, "ok": False}
            if decision.category == "chat":
                reason = classify_local_model_error(exc)
                web_sources = len(getattr(web_result_for_error, "hits", []) or [])
                result["error"]["message"] = reason
                if web_result_for_error is not None:
                    result["web_search"] = web_result_for_error.model_dump(mode="json")
                response_text = sanitize_assistant_identity(local_model_failure_message(reason, web_sources=web_sources))
            else:
                response_text = sanitize_assistant_identity(summarize_error(normalized, str(exc)))

        if not str(response_text or "").strip():
            response_text = "No obtuve una salida valida del sistema, pero sigo listo para responder."
        message = DesktopChatMessage(message_id=str(uuid4()), role="assistant", content=response_text, metadata={"result": result})
        return DesktopChatResponse(
            message=message,
            spoken_content=spoken_content_override if spoken_content_override is not None else self._spoken_variant(message.content),
            spoken_mode=spoken_mode,
            panel_snapshot=self._panels.compose(),
            raw_result=result,
        )

    def _handle_confirmation_reply(self, normalized: str) -> DesktopChatResponse:
        folded = self._intent_router._fold(normalized)  # noqa: SLF001
        result: dict = {}
        if self._is_negative_confirmation(folded):
            self.pending_action = None
            self.awaiting_confirmation = False
            message = DesktopChatMessage(message_id=str(uuid4()), role="assistant", content="Acción cancelada.", metadata={"result": result})
            return DesktopChatResponse(
                message=message,
                spoken_content=self._spoken_variant(message.content),
                spoken_mode="prepared",
                panel_snapshot=self._panels.compose(),
                raw_result=result,
            )
        if not self._is_affirmative_confirmation(folded):
            message = DesktopChatMessage(
                message_id=str(uuid4()),
                role="assistant",
                content="Hay una acción pendiente. Responde sí para ejecutarla o no para cancelarla.",
                metadata={"result": result},
            )
            return DesktopChatResponse(
                message=message,
                spoken_content=self._spoken_variant(message.content),
                spoken_mode="prepared",
                panel_snapshot=self._panels.compose(),
                raw_result=result,
            )

        pending = self.pending_action
        assert pending is not None
        self.pending_action = None
        self.awaiting_confirmation = False
        approved_decision = replace(pending.decision, approved=True)
        try:
            result, response_text, route_name = self._intent_router.execute(approved_decision)
            self._log_route(route_name, pending.prompt)
            self._log_execution(route_name, pending.prompt, result)
            response_text = f"Acceso concedido. {response_text}"
        except JarvisError as exc:
            result = {"error": exc.to_dict(), "ok": False}
            response_text = summarize_error(pending.prompt, str(exc))
        except Exception as exc:  # noqa: BLE001
            result = {"error": {"message": str(exc), "type": type(exc).__name__}, "ok": False}
            response_text = summarize_error(pending.prompt, str(exc))
        message = DesktopChatMessage(message_id=str(uuid4()), role="assistant", content=response_text, metadata={"result": result})
        return DesktopChatResponse(
            message=message,
            spoken_content=self._spoken_variant(message.content),
            spoken_mode="prepared",
            panel_snapshot=self._panels.compose(),
            raw_result=result,
        )

    def _spoken_variant(self, visual_text: str) -> str:
        spoken_text, _profile = prepare_spoken_response(
            visual_text,
            settings=self._settings,
            profile_name=self._settings.voice_profile_default,
        )
        return spoken_text

    def _build_model_request(self, normalized: str) -> tuple[ModelRequest, object | None]:
        lowered = normalized.casefold()
        is_coding = self._is_coding_query(lowered)
        logical_model = "coding_engine" if is_coding else "general_assistant"
        task_type = "coding" if is_coding else "assistant"
        required_capabilities = ("chat", "coding") if is_coding else ("chat",)
        web_result = None
        prompt = normalized
        if not is_coding and should_use_web_search(normalized, mode="auto"):
            provider = build_web_search_provider()
            candidate = provider.search(normalized, max_results=provider.status().max_results)
            if candidate.status == "ok" and candidate.hits:
                web_result = candidate
                prompt = build_grounded_web_prompt(
                    normalized,
                    candidate,
                    max_sources=self._settings.web_synthesis_max_sources,
                    snippet_chars=self._settings.web_synthesis_snippet_chars,
                )
            elif candidate.status in {"blocked", "unavailable", "disabled", "error", "empty"}:
                prompt = (
                    f"{normalized}\n\n"
                    f"Nota de Jarvis: busqueda web {candidate.status}. {candidate.message} "
                    "Responde con Ollama local, indicando claramente si no hay datos actuales disponibles."
                )
        profile = select_chat_generation_profile(normalized, self._settings, web_used=web_result is not None, is_coding=is_coding)
        return ModelRequest(
            prompt=prompt,
            messages=[
                {"role": "system", "content": jarvis_identity_prompt("Superficie: chat de escritorio local.")},
                {"role": "user", "content": prompt},
            ],
            logical_model=logical_model,
            task_type=task_type,
            required_capabilities=required_capabilities,
            temperature=profile.temperature,
            timeout_seconds=streaming_timeout_seconds(profile, self._settings),
            max_tokens=profile.max_tokens,
            metadata={"source": "desktop_chat", "generation_profile": profile.name, "context_profile": profile.context_profile},
        ), web_result

    def prepare_streaming_request(self, normalized: str) -> tuple[ModelRequest, object | None] | DesktopChatResponse | None:
        decision = self._intent_router.classify(normalized)
        if decision.category != "chat":
            return None
        if self._is_context_query(normalized.casefold()):
            web_status = build_web_search_provider().status()
            result = {"status": "ok", "context": "jarvis_context", "web_search": web_status.model_dump(mode="json")}
            message = DesktopChatMessage(message_id=str(uuid4()), role="assistant", content=self._context_summary(web_status), metadata={"result": result})
            return DesktopChatResponse(
                message=message,
                spoken_content=self._spoken_variant(message.content),
                spoken_mode="prepared",
                panel_snapshot=self._panels.compose(),
                raw_result=result,
            )
        return self._build_model_request(normalized)

    def _log_route(self, route: str, query: str) -> None:
        self._logger.info("desktop_chat_route", extra={"route": route, "query": query})

    def _log_execution(self, route: str, query: str, result: dict) -> None:
        self._logger.info(
            "desktop_chat_execution",
            extra={
                "route": route,
                "query": query,
                "status": result.get("status"),
                "target_window": self._target_window_from_result(result),
                "security_decision": self._security_decision_from_result(result),
            },
        )

    @staticmethod
    def _is_coding_query(lowered: str) -> bool:
        return any(keyword in lowered for keyword in ("codigo", "code", "python", "bug", "funcion", "function", "clase", "class", "script"))

    @staticmethod
    def _is_context_query(lowered: str) -> bool:
        return any(phrase in lowered for phrase in ("que contexto tienes", "qué contexto tienes", "que recuerdas", "qué recuerdas", "muestra tu contexto", "que sabes del proyecto", "qué sabes del proyecto"))

    @staticmethod
    def _context_summary(web_status) -> str:
        key_status = "configured true" if web_status.configured else "configured false"
        return (
            "Contexto actual de Jarvis\n\n"
            "Proyecto:\nJarvis desktop local en el workspace actual.\n\n"
            "Modelo:\nModo auto, usando Ollama local con gpt-oss:20b como cerebro principal.\n\n"
            "Web:\n"
            f"Brave Search provider={web_status.provider}, enabled={str(web_status.enabled).lower()}, API key {key_status}. "
            "Brave solo busca; Ollama local redacta.\n\n"
            "Reglas importantes:\n"
            "- Mi identidad es Jarvis, sin identidad externa.\n"
            "- OpenAI y Gemini estan bloqueados como modelos principales.\n"
            "- No mando secretos, archivos de entorno, tokens, PIN ni archivos privados a internet.\n"
            "- Patches y acciones sensibles requieren revision y confirmacion.\n"
            "- GitHub learning no clona repos sin confirmacion."
        )

    @staticmethod
    def _requires_confirmation(result: dict) -> bool:
        if result.get("confirmation_required"):
            return True
        ui = (result.get("verification_summary") or {}).get("ui") or {}
        return bool(ui.get("confirmation_required"))

    @staticmethod
    def _confirmation_prompt(result: dict) -> str:
        ui = (result.get("verification_summary") or {}).get("ui") or {}
        if result.get("confirmation_required") or ui.get("confirmation_required"):
            return "Esta aplicación requiere permiso. ¿Deseas continuar?"
        return str(result.get("message") or ui.get("message") or "Esta aplicación requiere permiso. ¿Deseas continuar?")

    @staticmethod
    def _target_window_from_result(result: dict) -> str | None:
        if (result.get("active_window") or {}).get("title"):
            return result["active_window"]["title"]
        if result.get("window_title"):
            return str(result["window_title"])
        ui = (result.get("verification_summary") or {}).get("ui") or {}
        if (ui.get("active_window") or {}).get("title"):
            return ui["active_window"]["title"]
        return None

    @staticmethod
    def _security_decision_from_result(result: dict) -> str | None:
        if result.get("security_decision"):
            return str(result["security_decision"])
        ui = (result.get("verification_summary") or {}).get("ui") or {}
        if ui.get("security_decision"):
            return str(ui["security_decision"])
        if result.get("confirmation_required") or ui.get("confirmation_required"):
            return "confirmation_required"
        return None

    @staticmethod
    def _is_affirmative_confirmation(folded: str) -> bool:
        return folded in {"si", "sí", "ok", "hazlo", "adelante", "confirmo", "continua", "ejecuta"}

    @staticmethod
    def _is_negative_confirmation(folded: str) -> bool:
        return folded in {"no", "cancela", "detente", "alto"}
