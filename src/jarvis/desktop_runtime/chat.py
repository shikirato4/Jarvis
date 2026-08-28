from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from uuid import uuid4

from jarvis.capability_manifest import get_capabilities_for_prompt, get_capabilities_summary, is_capability_query
from jarvis.config import Settings
from jarvis.conversation_context import ConversationContextBuilder, ContextFragment
from jarvis.agent_runtime import AgentController, AgentRunRequest, AgentRunStatus
from jarvis.chat_optimization import select_chat_generation_profile, streaming_timeout_seconds
from jarvis.conversational_safety import ConversationRisk, classify_conversation_risk, malware_safe_refusal
from jarvis.core.errors import JarvisError
from jarvis.identity import jarvis_identity_prompt, sanitize_assistant_identity
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.errors import (
    classify_provider_failure,
    provider_failure_message,
)
from jarvis.services import summarize_error, summarize_model_response
from jarvis.voice_runtime.spoken import prepare_spoken_response
from jarvis.web_search import build_grounded_web_prompt, build_web_search_provider, should_use_web_search
from jarvis.web_search.router import select_synthesis_hits, source_count_message

from .base import DesktopChatMessage, DesktopChatResponse
from .intent_router import DesktopIntentDecision, DesktopIntentRouter


@dataclass
class PendingDesktopAction:
    decision: DesktopIntentDecision
    prompt: str
    runtime_name: str | None = None
    target_window: str | None = None


class DesktopChatEngine:
    def __init__(
        self,
        bridge,
        panel_composer,
        action_executor,
        settings: Settings,
        *,
        context_builder: ConversationContextBuilder | None = None,
        agent_controller: AgentController | None = None,
    ) -> None:
        self._bridge = bridge
        self._panels = panel_composer
        self._actions = action_executor
        self._settings = settings
        self._context_builder = context_builder or ConversationContextBuilder()
        self._agent_controller = agent_controller
        self._intent_router = DesktopIntentRouter(bridge, action_executor)
        self._logger = logging.getLogger("jarvis.desktop.chat")
        self.pending_action: PendingDesktopAction | None = None
        self.awaiting_confirmation = False
        self._pending_agent_confirmation: tuple[str, str] | None = None

    def handle(
        self,
        text: str,
        *,
        request_transform: Callable[[ModelRequest], ModelRequest] | None = None,
        history: Iterable[ModelMessage] = (),
        tool_observations: Iterable[ModelMessage] = (),
        attachment_context: Iterable[ContextFragment] = (),
        memory_candidates: Iterable[ContextFragment] = (),
        correlation_id: str | None = None,
        request_finalized: Callable[[ModelRequest], None] | None = None,
    ) -> DesktopChatResponse:
        normalized = text.strip()
        result: dict = {}
        response_text = ""
        web_result_for_error = None
        spoken_content_override: str | None = None
        spoken_mode = "prepared"
        if self._pending_agent_confirmation is not None:
            return self._handle_agent_confirmation_reply(normalized)
        if self.awaiting_confirmation and self.pending_action is not None:
            return self._handle_confirmation_reply(normalized)
        risk = classify_conversation_risk(normalized)
        if risk == ConversationRisk.MALWARE_OFFENSIVE:
            return self._deterministic_response(
                malware_safe_refusal(),
                {"status": "blocked", "category": risk.value, "malware_safety": True, "clone_allowed": False},
                spoken_content="No puedo ayudar con uso de malware, pero te deje opciones defensivas en el chat.",
            )
        if is_capability_query(normalized):
            return self._deterministic_response(
                get_capabilities_summary(),
                {"status": "ok", "category": "capabilities", "agent_mode_real": True, "openai": "blocked", "gemini": "blocked"},
            )
        decision = self._intent_router.classify(normalized)
        self._logger.info(
            "desktop_intent_detected",
            extra={"intent": decision.category, "prompt_length": len(normalized)},
        )
        try:
            if decision.category == "chat":
                if self._is_context_query(normalized.casefold()):
                    web_status = build_web_search_provider().status()
                    result = {"status": "ok", "context": "jarvis_context", "web_search": web_status.model_dump(mode="json")}
                    response_text = self._context_summary(web_status)
                else:
                    request, web_result = self._build_model_request(normalized)
                    if request_transform is not None:
                        request = request_transform(request)
                    if correlation_id is not None:
                        request = request.model_copy(
                            update={"correlation_id": correlation_id}
                        )
                    request = self.finalize_model_request(
                        request,
                        history=history,
                        tool_observations=tool_observations,
                        attachment_context=attachment_context,
                        memory_candidates=memory_candidates,
                    )
                    if request_finalized is not None:
                        request_finalized(request)
                    web_result_for_error = web_result
                    route = f"models_runtime:{request.logical_model}"
                    if web_result is not None:
                        route = f"web_search:{web_result.provider}+{route}"
                    self._log_route(route, normalized)
                    if self._agent_controller is None:
                        model_response = self._bridge.runtime.infer_model(request)
                        result = model_response.model_dump(mode="json")
                        response_text = sanitize_assistant_identity(
                            summarize_model_response(result)
                        )
                    else:
                        agent_result = self._agent_controller.run(
                            AgentRunRequest(
                                user_message=normalized,
                                model_request=request,
                                max_steps=self._settings.agent_max_steps,
                                timeout_seconds=self._settings.agent_timeout_seconds,
                                correlation_id=correlation_id,
                                source_surface="desktop_chat",
                            )
                        )
                        result = agent_result.model_dump(mode="json")
                        self._merge_last_observation(result, agent_result)
                        if agent_result.observations and hasattr(self._panels, "invalidate"):
                            self._panels.invalidate()
                        response_text = self._agent_result_text(agent_result)
                        if (
                            agent_result.status
                            is AgentRunStatus.WAITING_CONFIRMATION
                            and agent_result.pending_confirmation is not None
                            and agent_result.pending_confirmation.confirmation_id
                        ):
                            self._pending_agent_confirmation = (
                                agent_result.run_id,
                                agent_result.pending_confirmation.confirmation_id,
                            )
                    if web_result is not None:
                        result["web_search"] = web_result.model_dump(mode="json")
            else:
                if decision.category == "research" and self._should_use_brave_for_research(normalized):
                    result, response_text, route_name = self._handle_current_research(normalized)
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
                failure = classify_provider_failure(exc)
                web_sources = len(getattr(web_result_for_error, "hits", []) or [])
                result["error"].update(
                    {"message": failure.detail, "code": failure.code, "web_sources": web_sources}
                )
                if web_result_for_error is not None:
                    result["web_search"] = web_result_for_error.model_dump(mode="json")
                response_text = sanitize_assistant_identity(
                    provider_failure_message(
                        failure,
                        selected_mode=self._settings.llm_mode,
                    )
                )
                if web_result_for_error is not None and web_sources:
                    response_text = (
                        "Encontré fuentes con Brave, pero Nemotron no pudo redactar el informe completo.\n\n"
                        + self._format_sources(web_result_for_error)
                        + "\n\n"
                        + response_text
                    )
            else:
                response_text = sanitize_assistant_identity(self._human_error_summary(normalized, str(exc)))

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

    def _handle_agent_confirmation_reply(
        self,
        normalized: str,
    ) -> DesktopChatResponse:
        assert self._pending_agent_confirmation is not None
        run_id, confirmation_id = self._pending_agent_confirmation
        folded = self._intent_router._fold(normalized)  # noqa: SLF001
        if self._is_negative_confirmation(folded):
            self._agent_controller.cancel_pending_confirmation(run_id)  # type: ignore[union-attr]
            self._pending_agent_confirmation = None
            return self._deterministic_response(
                "Acción cancelada.",
                {"run_id": run_id, "status": "cancelled"},
            )
        if not self._is_affirmative_confirmation(folded):
            return self._deterministic_response(
                "Hay una acción del agente pendiente. Responde sí para ejecutarla "
                "o no para cancelarla.",
                {"run_id": run_id, "status": "waiting_confirmation"},
            )
        self._pending_agent_confirmation = None
        agent_result = self._agent_controller.resume_confirmation(  # type: ignore[union-attr]
            run_id,
            confirmation_id,
        )
        result = agent_result.model_dump(mode="json")
        self._merge_last_observation(result, agent_result)
        if agent_result.observations and hasattr(self._panels, "invalidate"):
            self._panels.invalidate()
        message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="assistant",
            content=self._agent_result_text(agent_result),
            metadata={"result": result},
        )
        return DesktopChatResponse(
            message=message,
            spoken_content=self._spoken_variant(message.content),
            spoken_mode="prepared",
            panel_snapshot=self._panels.compose(),
            raw_result=result,
        )

    def _agent_result_text(self, agent_result) -> str:
        if agent_result.status is AgentRunStatus.COMPLETED:
            return sanitize_assistant_identity(agent_result.final_answer or "")
        if agent_result.status is AgentRunStatus.WAITING_CONFIRMATION:
            return (
                "La siguiente acción requiere confirmación explícita antes de "
                "continuar."
            )
        response_text = (
            "No pude completar el ciclo de agente de forma segura "
            f"(estado: {agent_result.status.value})."
        )
        partial_sources = self._format_agent_web_sources(agent_result.observations)
        if partial_sources:
            response_text = (
                f"{response_text}\n\n"
                "La síntesis final falló, pero estas fuentes ya habían sido "
                f"recuperadas:\n\n{partial_sources}"
            )
        return response_text

    @staticmethod
    def _merge_last_observation(result: dict, agent_result) -> None:
        """Keep stable public receipt fields while retaining the agent envelope."""
        if not agent_result.observations:
            return
        observation = agent_result.observations[-1]
        result.setdefault("tool_action", observation.action)
        result.setdefault("tool_status", observation.status.value)
        for key, value in observation.data.items():
            result.setdefault(key, value)

    def _spoken_variant(self, visual_text: str) -> str:
        spoken_text, _profile = prepare_spoken_response(
            visual_text,
            settings=self._settings,
            profile_name=self._settings.voice_profile_default,
        )
        return spoken_text

    def _deterministic_response(self, text: str, result: dict, *, spoken_content: str | None = None) -> DesktopChatResponse:
        message = DesktopChatMessage(message_id=str(uuid4()), role="assistant", content=text, metadata={"result": result})
        return DesktopChatResponse(
            message=message,
            spoken_content=spoken_content if spoken_content is not None else self._spoken_variant(message.content),
            spoken_mode="prepared",
            panel_snapshot=self._panels.compose(),
            raw_result=result,
        )

    def _handle_current_research(self, query: str) -> tuple[dict, str, str]:
        provider = build_web_search_provider()
        status = provider.status()
        if not status.available:
            result = {"status": "unavailable", "provider": status.provider, "message": status.message, "web_search": {"configured": status.configured}}
            return result, "No pude completar la busqueda web. Brave Search no esta disponible o no esta configurado.", "web_search.research"
        max_results = min(int(getattr(self._settings, "research_visible_sources", 5) or 5), 5)
        web_response = provider.search(query, max_results=max_results)
        result = {"status": web_response.status, "provider": web_response.provider, "web_search": web_response.model_dump(mode="json"), "partial_results": True}
        if web_response.status != "ok" or not web_response.hits:
            message = web_response.message or "No encontre resultados confiables con Brave."
            return result, f"No pude verificar esa noticia con fuentes confiables. {message}", "web_search.research"
        prompt = build_grounded_web_prompt(
            query,
            web_response,
            max_sources=self._settings.web_synthesis_max_sources,
            snippet_chars=self._settings.web_synthesis_snippet_chars,
        )
        profile = select_chat_generation_profile(query, self._settings, web_used=True, is_coding=False)
        request = ModelRequest(
            prompt=prompt,
            messages=[
                {"role": "system", "content": jarvis_identity_prompt("Research web con Brave. Responde breve, natural y con fuentes.")},
                {"role": "user", "content": prompt},
            ],
            logical_model="general_assistant",
            task_type="assistant",
            required_capabilities=("chat",),
            temperature=profile.temperature,
            timeout_seconds=streaming_timeout_seconds(profile, self._settings),
            max_tokens=min(profile.max_tokens, int(getattr(self._settings, "research_synthesis_max_tokens", 500) or 500)),
            metadata={"source": "desktop_research_web", "web_first": True},
        )
        try:
            model_response = self._bridge.runtime.infer_model(request)
            result["model"] = model_response.model_dump(mode="json")
            summary = sanitize_assistant_identity(summarize_model_response(result["model"]))
            return result, self._ensure_sources_visible(summary, web_response), "web_search.research+models_runtime"
        except Exception as exc:  # noqa: BLE001
            result["status"] = "partial"
            result["synthesis_error"] = classify_provider_failure(exc).code
            return (
                result,
                "Encontré fuentes con Brave, pero Nemotron no pudo redactar el informe completo.\n\n"
                + self._format_sources(web_response)
                + "\n\nPuedo reintentarlo con solo 2 o 3 fuentes para hacerlo mas rapido.",
                "web_search.research.partial",
            )

    def _ensure_sources_visible(self, summary: str, web_response) -> str:
        if "fuentes" in summary.casefold():
            return summary
        return f"{summary.rstrip()}\n\n{self._format_sources(web_response)}"

    def _format_sources(self, web_response) -> str:
        hits = select_synthesis_hits(
            web_response.hits,
            max_sources=min(int(getattr(self._settings, "research_visible_sources", 5) or 5), 5),
            snippet_chars=min(int(getattr(self._settings, "web_synthesis_snippet_chars", 500) or 500), 500),
        )
        lines = [source_count_message(len(web_response.hits), len(hits)), "", "Fuentes:"]
        for index, hit in enumerate(hits, start=1):
            snippet = " ".join((hit.snippet or "").split())[:280]
            lines.append(f"{index}. {hit.title} - {hit.source}\n   {snippet}\n   {hit.url}")
        return "\n".join(lines)

    @staticmethod
    def _format_agent_web_sources(observations) -> str:
        lines: list[str] = []
        seen_urls: set[str] = set()
        for observation in observations:
            if getattr(observation, "action", "") != "web.search":
                continue
            data = getattr(observation, "data", {}) or {}
            hits = data.get("hits", []) if isinstance(data, dict) else []
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                url = str(hit.get("url") or "").strip()
                if not url.startswith(("https://", "http://")) or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = str(hit.get("title") or "Fuente").strip() or "Fuente"
                source = str(hit.get("source") or "").strip()
                label = f"{title} — {source}" if source else title
                lines.append(f"- {label}\n  {url}")
                if len(lines) >= 5:
                    return "\n".join(lines)
        return "\n".join(lines)

    @staticmethod
    def _should_use_brave_for_research(text: str) -> bool:
        lowered = text.casefold()
        return should_use_web_search(text, mode="auto") or any(
            marker in lowered
            for marker in ("actual", "actualmente", "informe reciente", "que paso", "quÃ© paso", "noticia", "noticias", "reciente", "hoy", "fuentes")
        )

    @staticmethod
    def _human_error_summary(prompt: str, error: str) -> str:
        lowered = prompt.casefold()
        detail = error.strip()
        lowered_detail = detail.casefold()
        if "investiga" in lowered or "research" in lowered or "noticias" in lowered or "fuentes" in lowered:
            if "watchdog timeout" in lowered_detail or "operation deadline exceeded" in lowered_detail or "timed out" in lowered_detail:
                return (
                    "No pude terminar la investigacion a tiempo. Puede que la busqueda o la redaccion local hayan tardado demasiado. "
                    "Puedo intentarlo otra vez con menos fuentes o darte solo las fuentes encontradas."
                )
            return "No pude completar la investigacion en este momento. Intentalo de nuevo con una consulta mas especifica."
        return summarize_error(prompt, error)

    def _build_model_request(self, normalized: str) -> tuple[ModelRequest, object | None]:
        lowered = normalized.casefold()
        is_coding = self._is_coding_query(lowered)
        logical_model = "coding_engine" if is_coding else "general_assistant"
        task_type = "coding" if is_coding else "assistant"
        required_capabilities = ("chat", "coding") if is_coding else ("chat",)
        prompt = normalized
        profile = select_chat_generation_profile(
            normalized,
            self._settings,
            web_used=False,
            is_coding=is_coding,
        )
        context_profile = profile.context_profile
        prompt_context = get_capabilities_for_prompt(context_profile)
        system_context = "Superficie: chat de escritorio local."
        if prompt_context:
            system_context = f"{system_context}\n\nContexto activo de Jarvis:\n{prompt_context}"
        return ModelRequest(
            prompt=prompt,
            messages=[
                {"role": "system", "content": jarvis_identity_prompt(system_context)},
                {"role": "user", "content": prompt},
            ],
            logical_model=logical_model,
            task_type=task_type,
            required_capabilities=required_capabilities,
            temperature=profile.temperature,
            timeout_seconds=streaming_timeout_seconds(profile, self._settings),
            max_tokens=profile.max_tokens,
            metadata={
                "source": "desktop_chat",
                "generation_profile": profile.name,
                "context_profile": context_profile,
                "web_used": False,
            },
        ), None

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

    def finalize_model_request(
        self,
        request: ModelRequest,
        *,
        history: Iterable[ModelMessage] = (),
        tool_observations: Iterable[ModelMessage] = (),
        attachment_context: Iterable[ContextFragment] = (),
        memory_candidates: Iterable[ContextFragment] = (),
    ) -> ModelRequest:
        return self._context_builder.finalize_model_request(
            request,
            history=history,
            tool_observations=tool_observations,
            attachment_context=attachment_context,
            memory_candidates=memory_candidates,
            max_context_tokens=self._settings.conversation_context_max_tokens,
            safety_margin_tokens=(
                self._settings.conversation_context_safety_margin_tokens
            ),
        )

    def _log_route(self, route: str, query: str) -> None:
        self._logger.info(
            "desktop_chat_route",
            extra={"route": route, "query_length": len(query)},
        )

    def _log_execution(self, route: str, query: str, result: dict) -> None:
        self._logger.info(
            "desktop_chat_execution",
            extra={
                "route": route,
                "query_length": len(query),
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
        extra_phrases = (
            "contexto tienes",
            "que sabes",
            "que esta activo",
            "que modelo usa jarvis",
        )
        if any(phrase in lowered for phrase in extra_phrases):
            return True
        return any(phrase in lowered for phrase in ("que contexto tienes", "qué contexto tienes", "que recuerdas", "qué recuerdas", "muestra tu contexto", "que sabes del proyecto", "qué sabes del proyecto"))

    @staticmethod
    def _context_summary(web_status) -> str:
        key_status = "configured true" if web_status.configured else "configured false"
        return (
            "Contexto actual de Jarvis\n\n"
            "Proyecto:\nJarvis desktop local en el workspace actual.\n\n"
            "Modelo:\nNemotron 3 Nano Omni con modos AUTO, NVIDIA API y LOCAL. El objetivo local exacto es "
            "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4.\n\n"
            "Web:\n"
            f"Brave Search provider={web_status.provider}, enabled={str(web_status.enabled).lower()}, API key {key_status}. "
            "Brave solo busca; el proveedor Nemotron asignado redacta.\n\n"
            "Reglas importantes:\n"
            "- Mi identidad es Jarvis, sin identidad externa.\n"
            "- NVIDIA NIM es el proveedor remoto; el modo LOCAL no realiza trafico a NVIDIA.\n"
            "- OpenAI y Gemini estan bloqueados como proveedores principales.\n"
            "- No mando secretos, archivos de entorno, tokens, PIN ni archivos privados a internet.\n"
            "- Patches y acciones sensibles requieren revision y confirmacion.\n"
            "- GitHub learning no clona repos sin confirmacion.\n"
            "- Agent Mode real esta disponible en Guided Control con Confirm Action para acciones sensibles.\n\n"
            "Funciones reales disponibles:\n"
            f"{get_capabilities_for_prompt('capability')}"
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
