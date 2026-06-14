from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from jarvis.actions.models import ActionExecutionReceipt, ActionStep, ExecutionStatus
from jarvis.actions.router import ActionRouter
from jarvis.cognition.context import RetrievedContextFormatter
from jarvis.core.capabilities import CapabilityRegistry, RegisteredCapability
from jarvis.core.errors import OrchestrationError
from jarvis.identity import jarvis_identity_prompt, sanitize_assistant_identity
from jarvis.memory.service import MemoryService
from jarvis.memory_semantic.base import RetrievedContext, SemanticSearchQuery
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.memory_semantic.service import SemanticMemoryService
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.models_runtime.service import ModelService
from jarvis.ui_automation.base import UIAutomationMode, WriteTextRequest
from jarvis.ui_automation.service import UIAutomationService
from jarvis.voice_runtime.service import VoiceRuntimeService

from .models import OrchestrationRequest, OrchestrationResponse


class CognitiveOrchestrator:
    def __init__(
        self,
        router: ActionRouter,
        memory: MemoryService,
        capabilities: CapabilityRegistry,
        models: ModelService,
        semantic_memory: SemanticMemoryService | None = None,
        context_formatter: RetrievedContextFormatter | None = None,
        max_context_chunks: int = 8,
        ui_automation: UIAutomationService | None = None,
        voice_runtime: VoiceRuntimeService | None = None,
    ) -> None:
        self._router = router
        self._memory = memory
        self._capabilities = capabilities
        self._models = models
        self._semantic_memory = semantic_memory
        self._context_formatter = context_formatter or RetrievedContextFormatter(char_budget=4000)
        self._max_context_chunks = max_context_chunks
        self._ui_automation = ui_automation
        self._voice_runtime = voice_runtime
        self._logger = logging.getLogger("jarvis.cognition.orchestrator")

    def handle(self, request: OrchestrationRequest) -> OrchestrationResponse:
        correlation_id = str(uuid4())
        resolved_intent = request.intent or self._infer_intent(request.query or "", request.payload, correlation_id=correlation_id)
        if resolved_intent == "general_chat":
            return self._handle_general_chat(correlation_id, request)
        capability = self._capabilities.get(resolved_intent)
        if capability is None:
            raise OrchestrationError(f"unsupported intent '{resolved_intent}'")

        if request.persist_input and request.query:
            self._memory.store_memory(
                kind="interaction.input",
                content=request.query,
                source="user",
                metadata={"correlation_id": correlation_id, "intent": resolved_intent},
            )

        if request.plan:
            receipts = self._router.execute_plan(
                request.plan,
                correlation_id=correlation_id,
                metadata={"intent": resolved_intent},
            )
            return OrchestrationResponse(
                correlation_id=correlation_id,
                resolved_intent=resolved_intent,
                plan=request.plan,
                receipts=receipts,
            )

        semantic_context = self._retrieve_semantic_context(capability, request, correlation_id=correlation_id)

        if capability.descriptor.intent == "document_ingest":
            return self._handle_document_ingest(correlation_id, request)
        if capability.descriptor.intent == "semantic_search":
            return self._handle_semantic_search(correlation_id, request)
        if capability.descriptor.intent == "research_brief":
            return self._handle_research_brief(correlation_id, request, capability, semantic_context)
        if capability.descriptor.intent == "contextual_writing":
            return self._handle_contextual_writing(correlation_id, request, semantic_context)

        plan = self._resolve_plan(capability, request, correlation_id=correlation_id, semantic_context=semantic_context)
        receipts = self._router.execute_plan(
            plan,
            correlation_id=correlation_id,
            metadata={"intent": resolved_intent},
        )
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent=resolved_intent,
            plan=plan,
            receipts=receipts,
        )

    def _resolve_plan(
        self,
        capability: RegisteredCapability,
        request: OrchestrationRequest,
        *,
        correlation_id: str,
        semantic_context: RetrievedContext | None,
    ) -> list[ActionStep]:
        generated_plan = self._generate_model_plan(
            capability,
            request,
            correlation_id=correlation_id,
            semantic_context=semantic_context,
        )
        if generated_plan:
            return generated_plan
        plan = self._capabilities.build_plan(capability.descriptor.intent, request)
        if semantic_context and plan and plan[0].action == "writer.compose_note":
            enriched = dict(plan[0].payload)
            enriched["findings"] = self._merge_findings(enriched.get("findings", []), semantic_context)
            enriched.setdefault("references", semantic_context.sources)
            plan[0] = ActionStep(action=plan[0].action, payload=enriched)
        return plan

    def _generate_model_plan(
        self,
        capability: RegisteredCapability,
        request: OrchestrationRequest,
        *,
        correlation_id: str,
        semantic_context: RetrievedContext | None,
    ) -> list[ActionStep] | None:
        if not capability.descriptor.supports_planning or not request.query:
            return None
        context_block = self._context_formatter.format_for_prompt(semantic_context) if semantic_context else ""
        try:
            response = self._models.infer(
                ModelRequest(
                    task_type="planning",
                    logical_model="planner",
                    required_capabilities=("planning",),
                    correlation_id=correlation_id,
                    messages=[
                        ModelMessage(
                            role="system",
                            content=(
                                "Return a JSON array of action steps. "
                                f"Allowed actions: {', '.join(capability.descriptor.action_names)}. "
                                "Each item must have keys 'action' and 'payload'. "
                                f"{context_block}"
                            ),
                        ),
                        ModelMessage(role="user", content=request.query),
                    ],
                )
            )
            raw_steps = json.loads(response.content)
            if not isinstance(raw_steps, list):
                return None
            steps = [ActionStep.model_validate(item) for item in raw_steps]
            if all(step.action in capability.descriptor.action_names for step in steps):
                return steps
        except Exception as exc:
            self._log_suppressed_exception(
                "orchestrator_model_plan_fallback",
                exc,
                correlation_id=correlation_id,
                intent=capability.descriptor.intent,
            )
            return None
        return None

    def _handle_research_brief(
        self,
        correlation_id: str,
        request: OrchestrationRequest,
        capability: RegisteredCapability,
        semantic_context: RetrievedContext | None,
    ) -> OrchestrationResponse:
        plan = self._generate_model_plan(capability, request, correlation_id=correlation_id, semantic_context=semantic_context)
        if not plan:
            plan = self._capabilities.build_plan(capability.descriptor.intent, request)
        research_payload = dict(plan[0].payload)
        research_receipt = self._router.execute(
            plan[0].action,
            research_payload,
            correlation_id=correlation_id,
            metadata={"intent": "research_brief", "stage": "research"},
        )
        findings = [hit["snippet"] for hit in research_receipt.data.get("hits", [])]
        references = [hit["path"] for hit in research_receipt.data.get("hits", [])]
        if semantic_context:
            findings = findings + self._context_formatter.findings(semantic_context)
            references = list(dict.fromkeys(references + semantic_context.sources))
        summary_findings = self._summarize_findings(findings, request, correlation_id=correlation_id)
        writer_payload = dict(plan[1].payload)
        writer_payload["findings"] = summary_findings or findings
        writer_payload["references"] = references
        writer_receipt = self._router.execute(
            plan[1].action,
            writer_payload,
            correlation_id=correlation_id,
            metadata={"intent": "research_brief", "stage": "write"},
        )
        final_plan = [plan[0], ActionStep(action=plan[1].action, payload=writer_payload)]
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent="research_brief",
            plan=final_plan,
            receipts=[research_receipt, writer_receipt],
        )

    def _summarize_findings(self, findings: list[str], request: OrchestrationRequest, *, correlation_id: str) -> list[str]:
        if not findings:
            return []
        try:
            response = self._models.infer(
                ModelRequest(
                    task_type="summarization",
                    logical_model="summarizer",
                    required_capabilities=("summarization",),
                    correlation_id=correlation_id,
                    messages=[
                        ModelMessage(
                            role="system",
                            content="Summarize the evidence into a JSON array of concise findings. Return JSON only.",
                        ),
                        ModelMessage(role="user", content="\n".join(findings[:20])),
                    ],
                    metadata={"intent": request.intent or "research_brief"},
                )
            )
            parsed = json.loads(response.content)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception as exc:
            self._log_suppressed_exception(
                "orchestrator_findings_summary_fallback",
                exc,
                correlation_id=correlation_id,
                intent=request.intent or "research_brief",
            )
            return findings
        return findings

    def _infer_intent(self, query: str, payload: dict[str, Any], *, correlation_id: str) -> str:
        heuristic_intent = self._infer_heuristic_intent(query, payload)
        if heuristic_intent is not None:
            self._logger.info(
                "orchestrator_intent_heuristic",
                extra={"correlation_id": correlation_id, "intent": heuristic_intent, "query": query},
            )
            return heuristic_intent

        keyword_intent = self._capabilities.infer_intent(query, default_intent=None)
        if keyword_intent is not None and self._is_safe_keyword_intent(keyword_intent, query):
            return keyword_intent

        if self._looks_like_general_chat(query):
            self._logger.info("orchestrator_intent_fallback", extra={"correlation_id": correlation_id, "intent": "general_chat"})
            return "general_chat"
        try:
            response = self._models.infer(
                ModelRequest(
                    task_type="classification",
                    logical_model="general_assistant",
                    required_capabilities=("classification",),
                    correlation_id=correlation_id,
                    messages=[
                        ModelMessage(
                            role="system",
                            content=(
                                "Classify the user request into exactly one intent from this list: "
                                + ", ".join(item.descriptor.intent for item in self._capabilities.list_capabilities())
                                + ". Return JSON with key 'intent'."
                            ),
                        ),
                        ModelMessage(role="user", content=query),
                    ],
                )
            )
            payload = json.loads(response.content)
            if isinstance(payload, dict):
                intent = payload.get("intent")
                valid_intents = {item.descriptor.intent for item in self._capabilities.list_capabilities()}
                if intent in valid_intents and self._is_safe_keyword_intent(str(intent), query):
                    return str(intent)
        except Exception as exc:
            self._log_suppressed_exception(
                "orchestrator_model_classification_fallback",
                exc,
                correlation_id=correlation_id,
                query=query,
            )
        inferred = self._capabilities.infer_intent(query, default_intent=None)
        if inferred is not None and self._is_safe_keyword_intent(inferred, query):
            return inferred
        if inferred is None and self._looks_like_explicit_research(query):
            return "research_brief"
        if inferred is None and self._looks_like_science(query):
            return "science"
        if inferred is None:
            return "general_chat"
        if inferred is None:
            raise OrchestrationError("unable to infer intent")
        return inferred

    def _handle_general_chat(self, correlation_id: str, request: OrchestrationRequest) -> OrchestrationResponse:
        prompt = (request.query or request.payload.get("query") or "").strip()
        if not prompt:
            raise OrchestrationError("general chat requires a prompt")
        logical_model = "coding_engine" if self._looks_like_coding_query(prompt) else "general_assistant"
        task_type = "coding" if logical_model == "coding_engine" else "assistant"
        required_capabilities = ("chat", "coding") if logical_model == "coding_engine" else ("chat",)
        response = self._models.infer(
            ModelRequest(
                prompt=prompt,
                logical_model=logical_model,
                task_type=task_type,
                required_capabilities=required_capabilities,
                correlation_id=correlation_id,
                messages=[
                    ModelMessage(
                        role="system",
                        content=jarvis_identity_prompt(
                            "Responde directamente. No menciones runtimes internos ni busqueda de workspace salvo que el usuario lo pida."
                        ),
                    ),
                    ModelMessage(role="user", content=prompt),
                ],
                metadata={"intent": "general_chat", "surface": request.payload.get("surface", "runtime")},
            )
        )
        self._logger.info(
            "orchestrator_general_chat",
            extra={
                "correlation_id": correlation_id,
                "logical_model": response.logical_model,
                "model_name": response.model_name,
                "fallback_used": response.fallback_used,
            },
        )
        response_data = response.model_dump(mode="json")
        response_data["content"] = sanitize_assistant_identity(str(response_data.get("content") or ""))
        receipt = ActionExecutionReceipt(
            correlation_id=correlation_id,
            action="model.chat",
            status=ExecutionStatus.SUCCESS,
            message="general chat completed",
            data=response_data,
        )
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent="general_chat",
            plan=[ActionStep(action="model.chat", payload={"prompt": prompt, "logical_model": logical_model})],
            receipts=[receipt],
        )

    def _handle_document_ingest(self, correlation_id: str, request: OrchestrationRequest) -> OrchestrationResponse:
        if self._semantic_memory is None:
            raise OrchestrationError("semantic memory service is not available")
        document = self._semantic_memory.ingest_document(DocumentIngestionRequest.model_validate(request.payload))
        receipt = ActionExecutionReceipt(
            correlation_id=correlation_id,
            action="semantic.ingest",
            status=ExecutionStatus.SUCCESS,
            message="semantic document ingested",
            data=document.model_dump(mode="json"),
        )
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent="document_ingest",
            plan=[ActionStep(action="semantic.ingest", payload=request.payload)],
            receipts=[receipt],
        )

    def _handle_semantic_search(self, correlation_id: str, request: OrchestrationRequest) -> OrchestrationResponse:
        if self._semantic_memory is None:
            raise OrchestrationError("semantic memory service is not available")
        query = SemanticSearchQuery(
            query=request.payload.get("query", request.query or ""),
            collection_name=request.payload.get("collection_name"),
            top_k=request.payload.get("top_k"),
            min_score=request.payload.get("min_score"),
            source_types=tuple(request.payload.get("source_types", ())),
            metadata_filters=request.payload.get("metadata_filters", {}),
            correlation_id=correlation_id,
        )
        result = self._semantic_memory.search(query)
        receipt = ActionExecutionReceipt(
            correlation_id=correlation_id,
            action="semantic.search",
            status=ExecutionStatus.SUCCESS,
            message="semantic search completed",
            data=result.model_dump(mode="json"),
        )
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent="semantic_search",
            plan=[ActionStep(action="semantic.search", payload=query.model_dump(mode="json"))],
            receipts=[receipt],
        )

    def _handle_contextual_writing(
        self,
        correlation_id: str,
        request: OrchestrationRequest,
        semantic_context: RetrievedContext | None,
    ) -> OrchestrationResponse:
        findings = self._merge_findings(request.payload.get("findings", []), semantic_context)
        payload = dict(request.payload)
        payload.setdefault("title", "Contextual draft")
        payload.setdefault("objective", request.query or "Write using recovered context")
        payload["findings"] = findings
        payload.setdefault("references", semantic_context.sources if semantic_context else [])
        receipt = self._router.execute(
            "writer.compose_note",
            payload,
            correlation_id=correlation_id,
            metadata={"intent": "contextual_writing"},
        )
        receipts = [receipt]
        if self._ui_automation and payload.get("delivery_mode") in {UIAutomationMode.COPILOT.value, UIAutomationMode.DIRECT.value}:
            ui_receipt = self._ui_automation.write_text(
                WriteTextRequest(
                    text=receipt.data.get("content", ""),
                    mode=UIAutomationMode(payload.get("delivery_mode", UIAutomationMode.COPILOT.value)),
                    focus_target=payload.get("target_window"),
                    ensure_window_contains=payload.get("ensure_window_contains"),
                ),
                correlation_id=correlation_id,
            )
            receipts.append(
                ActionExecutionReceipt(
                    correlation_id=correlation_id,
                    action="interface.write_text",
                    status=ExecutionStatus.SUCCESS,
                    message=ui_receipt.message,
                    data=ui_receipt.model_dump(mode="json"),
                )
            )
        if self._voice_runtime and payload.get("response_mode") in {"voice", "both"}:
            voice_receipt = self._voice_runtime.speak(receipt.data.get("content", ""), correlation_id=correlation_id)
            receipts.append(
                ActionExecutionReceipt(
                    correlation_id=correlation_id,
                    action="voice_runtime.speak",
                    status=ExecutionStatus.SUCCESS,
                    message=voice_receipt.message,
                    data=voice_receipt.model_dump(mode="json"),
                )
            )
        return OrchestrationResponse(
            correlation_id=correlation_id,
            resolved_intent="contextual_writing",
            plan=[ActionStep(action="writer.compose_note", payload=payload)],
            receipts=receipts,
        )

    def _retrieve_semantic_context(
        self,
        capability: RegisteredCapability,
        request: OrchestrationRequest,
        *,
        correlation_id: str,
    ) -> RetrievedContext | None:
        if self._semantic_memory is None or not request.query or request.payload.get("disable_semantic_context"):
            return None
        if capability.descriptor.intent in {
            "vision",
            "screen_read",
            "ui_awareness",
            "voice",
            "operate",
            "remember",
            "document_ingest",
            "system_open",
            "system_search",
            "ui_control",
            "direct_write",
            "unity",
        }:
            return None
        try:
            context = self._semantic_memory.retrieve_context(
                SemanticSearchQuery(
                    query=request.payload.get("semantic_query", request.query),
                    collection_name=request.payload.get("collection_name"),
                    top_k=request.payload.get("top_k"),
                    min_score=request.payload.get("min_score"),
                    source_types=tuple(request.payload.get("source_types", ())),
                    metadata_filters=request.payload.get("metadata_filters", {}),
                    correlation_id=correlation_id,
                )
            )
            if not context.chunks:
                return None
            if len(context.chunks) > 0:
                context.chunks = context.chunks[: self._max_context_chunks]
            return context
        except Exception as exc:
            self._log_suppressed_exception(
                "orchestrator_semantic_context_skipped",
                exc,
                correlation_id=correlation_id,
                intent=capability.descriptor.intent,
            )
            return None

    def _merge_findings(self, existing: list[str], semantic_context: RetrievedContext | None) -> list[str]:
        findings = [str(item) for item in existing]
        if semantic_context:
            findings.extend(self._context_formatter.findings(semantic_context))
        return list(dict.fromkeys(findings))

    def _context_limit(self) -> int:
        return self._max_context_chunks

    @staticmethod
    def _looks_like_general_chat(query: str) -> bool:
        lowered = query.casefold().strip()
        if not lowered:
            return True
        research_keywords = ("investiga", "investigar", "research", "evidencia", "fuentes", "workspace", "documenta")
        action_keywords = (
            "abre ",
            "open ",
            "busca ",
            "buscar ",
            "search ",
            "find ",
            "encuentra ",
            "ejecuta ",
            "run ",
            "lee la pantalla",
            "describe la pantalla",
            "que hay en mi pantalla",
            "que hay en pantalla",
            "/",
        )
        if any(keyword in lowered for keyword in research_keywords):
            return False
        if lowered.startswith(("busca ", "buscar ", "search ", "find ")) and any(
            marker in lowered
            for marker in (
                "significado de",
                "meaning of",
                "definicion de",
                "definición de",
                "que es",
                "qué es",
                "what is",
            )
        ):
            return True
        if lowered.startswith(action_keywords):
            return False
        return True

    @staticmethod
    def _looks_like_explicit_research(query: str) -> bool:
        lowered = query.casefold()
        direct_research_verbs = ("investiga", "investigar", "research")
        scoped_research_verbs = ("analiza", "compara", "documenta")
        research_scope = ("fuentes", "evidencia", "evidence", "workspace", "repositorio", "repo", "documentacion", "docs")
        return any(keyword in lowered for keyword in direct_research_verbs) or (
            any(keyword in lowered for keyword in scoped_research_verbs)
            and any(keyword in lowered for keyword in research_scope)
        ) or (
            any(keyword in lowered for keyword in research_scope) and any(token in lowered for token in ("con ", "usando ", "sobre ", "de "))
        )

    @staticmethod
    def _looks_like_semantic_search_request(query: str) -> bool:
        lowered = query.casefold()
        semantic_terms = ("semant", "semantic", "embedding", "indice", "indexado", "coleccion", "collection", "corpus")
        retrieval_terms = ("busca", "buscar", "recupera", "retrieve", "search", "encuentra")
        return any(term in lowered for term in semantic_terms) and any(term in lowered for term in retrieval_terms)

    @staticmethod
    def _looks_like_science(query: str) -> bool:
        lowered = query.casefold()
        return any(keyword in lowered for keyword in ("calcula", "deriv", "integr", "simula", "resuelve", "ecuacion"))

    @staticmethod
    def _looks_like_security(query: str) -> bool:
        lowered = query.casefold()
        return any(keyword in lowered for keyword in ("seguridad", "vulnerabilidad", "password", "contrasena", "contraseña", "cve", "xss", "sqli"))

    @staticmethod
    def _looks_like_system_open(query: str) -> bool:
        lowered = query.casefold().strip()
        if not lowered.startswith(("abre ", "abrir ", "open ", "launch ")):
            return False
        compound_markers = (" y busca ", " y search ", " y escribe ", " and search ", " and type ", " paso a paso")
        return not any(marker in lowered for marker in compound_markers)

    @staticmethod
    def _looks_like_system_search(query: str) -> bool:
        lowered = query.casefold().strip()
        return lowered.startswith(("busca ", "buscar ", "search ", "find ", "encuentra ")) and any(
            marker in lowered for marker in ("sistema", "system", "archivo", "file", "carpeta", "folder", ".")
        )

    @staticmethod
    def _looks_like_visual_request(query: str) -> bool:
        lowered = query.casefold()
        if CognitiveOrchestrator._looks_like_metaphorical_visual_request(lowered):
            return False
        visual_markers = (
            "pantalla",
            "screen",
            "escritorio",
            "desktop",
            "ventana",
            "window",
            "imagen",
            "image",
            "captura",
            "screenshot",
            "ocr",
            "visible",
            "abierto",
        )
        visual_verbs = (
            "lee",
            "leer",
            "describe",
            "mira",
            "ve",
            "ves",
            "ver",
            "localiza",
            "encuentra",
            "detecta",
            "captura",
            "analiza",
            "analizar",
            "que ves",
            "que hay",
            "dice",
        )
        return any(marker in lowered for marker in visual_markers) and any(verb in lowered for verb in visual_verbs)

    @staticmethod
    def _looks_like_ui_awareness_request(query: str) -> bool:
        lowered = query.casefold()
        grounding_terms = ("boton", "botón", "texto", "campo", "elemento", "button", "text", "field", "element")
        return CognitiveOrchestrator._looks_like_visual_request(query) and any(term in lowered for term in grounding_terms)

    @staticmethod
    def _looks_like_metaphorical_visual_request(lowered: str) -> bool:
        return any(
            phrase in lowered
            for phrase in (
                "ves lo que digo",
                "ves lo que quiero decir",
                "ves por donde voy",
                "ves mi punto",
                "do you see what i mean",
            )
        )

    @staticmethod
    def _looks_like_voice_request(query: str) -> bool:
        lowered = query.casefold()
        return any(keyword in lowered for keyword in ("dicta", "dictado", "transcribe", "escucha", "listen", "transcription"))

    def _infer_heuristic_intent(self, query: str, payload: dict[str, Any]) -> str | None:
        if payload.get("image_path"):
            return "vision"
        if payload.get("audio_path"):
            return "voice_runtime"
        if payload.get("command"):
            return "operate"
        if self._looks_like_system_open(query):
            return "system_open"
        if self._looks_like_system_search(query):
            return "system_search"
        if self._looks_like_ui_awareness_request(query):
            return "ui_awareness"
        if self._looks_like_visual_request(query):
            return "screen_read"
        if self._looks_like_voice_request(query):
            return "voice_runtime"
        if self._looks_like_explicit_research(query):
            return "research_brief"
        if self._looks_like_desktop_agent_task(query):
            return "desktop_agent"
        if self._looks_like_security(query):
            return "security"
        if self._looks_like_science(query):
            return "science"
        return None

    def _is_safe_keyword_intent(self, intent: str, query: str) -> bool:
        if intent == "desktop_agent":
            return self._looks_like_desktop_agent_task(query)
        if intent in {"vision", "screen_read"}:
            return self._looks_like_visual_request(query)
        if intent == "ui_awareness":
            return self._looks_like_ui_awareness_request(query)
        if intent in {"research", "deep_research", "research_brief"}:
            return self._looks_like_explicit_research(query)
        if intent == "semantic_search":
            return self._looks_like_semantic_search_request(query)
        if intent == "system_search":
            return self._looks_like_system_search(query)
        if intent == "system_open":
            return self._looks_like_system_open(query)
        if intent == "voice_runtime":
            return self._looks_like_voice_request(query)
        return True

    @staticmethod
    def _looks_like_coding_query(query: str) -> bool:
        lowered = query.casefold()
        return any(keyword in lowered for keyword in ("codigo", "code", "python", "bug", "funcion", "function", "clase", "class", "script"))

    @classmethod
    def _looks_like_desktop_agent_task(cls, query: str) -> bool:
        lowered = query.casefold().strip()
        if not lowered:
            return False
        if cls._looks_like_explicit_research(lowered) or cls._looks_like_science(lowered) or cls._looks_like_security(lowered):
            return False
        direct_markers = (
            "haz click",
            "haz clic",
            "click en",
            "clic en",
            "ve a la ventana activa",
            "ventana activa",
            "llena este formulario",
            "rellena este formulario",
            "completa este formulario",
            "cambia a ",
            "switch to ",
            "guarda el documento",
            "guardar el documento",
            "continua mi libro",
            "continua el documento",
            "continúa mi libro",
            "continúa el documento",
            "escribe esto",
            "escribir esto",
            "mueve el mouse",
            "scroll",
            "desplaza",
            "explorador",
            "explorer",
        )
        if any(marker in lowered for marker in direct_markers):
            return True
        if any(token in lowered for token in ("archivo", "carpeta")) and any(
            marker in lowered for marker in ("abre", "abrir", "crea", "crear", "copia", "copiar", "mueve", "mover", "renombra", "renombrar", "busca", "buscar")
        ):
            return True
        app_markers = ("chrome", "word", "vscode", "explorador", "explorer", "youtube")
        return lowered.startswith(("abre ", "abrir ", "open ", "launch ")) and any(marker in lowered for marker in app_markers)

    def _log_suppressed_exception(self, event: str, exc: Exception, **extra: Any) -> None:
        self._logger.warning(
            event,
            extra={
                **extra,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
        )
