from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock, RLock, Timer
from time import perf_counter
from typing import Any
from uuid import uuid4

from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.attachments import enrich_model_request
from jarvis.attachments.pipeline import ProcessingMode
from jarvis.conversation_context import (
    ConversationContextBuilder,
    context_fragments_from_retrieved_context,
)
from jarvis.environment import detect_environment
from .actions import DesktopQuickActionExecutor, build_quick_actions
from .base import (
    DesktopChatMessage,
    DesktopPanelSnapshot,
    DesktopShellState,
    DesktopVoiceState,
)
from .bridge import DesktopRuntimeBridge
from .chat import DesktopChatEngine
from .panels import DesktopPanelComposer
from .progress import project_runtime_progress
from .voice import DesktopVoiceController
from jarvis.image_runtime.presentation import (
    image_job_content,
    reconcile_image_job_message,
)
from jarvis.models_runtime.errors import (
    classify_provider_failure,
    provider_failure_message,
)
from jarvis.models_runtime.qwen_identity import local_model_policy_description
from jarvis.models_runtime.hybrid import RequestExecutionContext
from jarvis.models_runtime.base import ModelMessage, ModelRequest
from jarvis.reasoning import ReasoningPreference
from jarvis.persistent_config import load_persistent_config, save_persistent_config
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.local_runtime import LocalLaunchProfile, LocalRuntimeIncompatibleError
from jarvis.services import (
    summarize_autonomy_view,
    summarize_research_task,
    summarize_system_operation,
    summarize_writing_receipt,
)
from jarvis.web_search import SearchMode, build_web_search_provider, decide_search_mode


class DesktopRuntimeService:
    _PANEL_CACHE_TTL_SECONDS = 60.0
    _DEV_RUNTIME_CACHE_TTL_SECONDS = 1.0
    _VOICE_CACHE_TTL_SECONDS = 0.25

    def __init__(self, jarvis_app) -> None:
        self._jarvis = jarvis_app
        self._bridge = DesktopRuntimeBridge(jarvis_app)
        self._panels = DesktopPanelComposer(self._bridge)
        self._actions = DesktopQuickActionExecutor(self._bridge)
        event_sink = getattr(getattr(jarvis_app, "event_bus", None), "publish", None)
        event_bus = getattr(jarvis_app, "event_bus", None)
        self._context_builder = ConversationContextBuilder(event_sink=event_sink)
        self._chat = DesktopChatEngine(
            self._bridge,
            self._panels,
            self._actions,
            jarvis_app.settings,
            context_builder=self._context_builder,
            agent_controller=jarvis_app.agent_controller,
            web_research=jarvis_app.web_research_orchestrator,
        )
        self._voice = DesktopVoiceController(
            jarvis_app.voice_runtime_service,
            jarvis_app.settings,
            self._handle_voice_command,
        )
        self._logger = logging.getLogger("jarvis.desktop.service")
        self._state_lock = RLock()
        self._request_lock = Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="jarvis-desktop"
        )
        self._quick_actions = build_quick_actions()
        self._request_futures: dict[str, Future] = {}
        self._completed_chat_requests: dict[str, object] = {}
        self._active_stream_id: str | None = None
        self._active_stream_cancel: Event | None = None
        self._active_research_cancel: Event | None = None
        self._progress_message_by_correlation: dict[str, str] = {}
        self._conversation: list[DesktopChatMessage] = [
            DesktopChatMessage(
                message_id=str(uuid4()),
                role="system",
                content="JARVIS desktop interface online. All runtimes remain delegated to the existing backend.",
            )
        ]
        self._model_context_by_message_id: dict[str, list[ModelMessage]] = {}
        self._busy = False
        self._activity_label = "STARTING" if not jarvis_app.started else "IDLE"
        self._performance: dict[str, float | str | int | None] = {
            "startup_ms": None,
            "last_request_ms": None,
            "model_latency_ms": None,
            "last_first_feedback_ms": 0.0,
            "last_shell_state_ms": None,
            "last_quick_action_ms": None,
            "requests_completed": 0,
            "ui_refresh_applied": 0,
            "ui_refresh_skipped": 0,
        }
        self._panel_snapshot_cache: DesktopPanelSnapshot | None = None
        self._panel_snapshot_cached_at = 0.0
        self._dev_runtime_cache: dict[str, Any] | None = None
        self._dev_runtime_cached_at = 0.0
        self._voice_cache = None
        self._voice_cached_at = 0.0
        self._code_agent: CodeAgentRuntimeService | None = None
        self._last_dev_result: dict[str, Any] = {
            "status": "idle",
            "action": "none",
            "summary": "Code Agent listo. Genera patches revisables; nada se aplica automaticamente.",
        }
        self._research_progress: dict[str, Any] = {}
        self._attachment_request_id = f"desktop-draft-{uuid4()}"
        if event_bus is not None and hasattr(event_bus, "subscribe"):
            event_bus.subscribe("image.generation_progress", self._on_image_progress)
        if event_bus is not None and hasattr(event_bus, "subscribe_all"):
            event_bus.subscribe_all(self._on_runtime_activity)

    def _on_runtime_activity(
        self,
        event_name: str,
        payload: dict[str, Any],
    ) -> None:
        """Project the canonical runtime activity stream into live Thinking."""

        if event_name == "deep_research.progress":
            correlation_id = str(payload.get("correlation_id") or "").strip()
            with self._state_lock:
                self._research_progress = dict(payload)
                message_id = self._progress_message_by_correlation.get(correlation_id)
            if message_id:
                metrics = (
                    f"{int(payload.get('sources_discovered') or 0)} sources found · "
                    f"{int(payload.get('sources_opened') or 0)} reviewed · "
                    f"{int(payload.get('domains') or 0)} domains"
                )
                self._replace_conversation_message(
                    message_id,
                    self._conversation_message_content(message_id),
                    metadata={
                        "thinking": {
                            "status": "researching",
                            "lines": [str(payload.get("message") or "Researching"), metrics],
                        }
                    },
                )
            return

        if not event_name.startswith(
            (
                "agent_controller.",
                "action_executor.",
                "terminal.",
                "model.auto_continuation_",
            )
        ):
            return
        correlation_id = str(payload.get("correlation_id") or "").strip()
        if not correlation_id:
            return
        with self._state_lock:
            message_id = self._progress_message_by_correlation.get(correlation_id)
        if not message_id:
            return
        snapshot = self._jarvis.runtime_activity_store.snapshot(
            correlation_id=correlation_id
        )
        projection = project_runtime_progress(snapshot)
        self._replace_conversation_message(
            message_id,
            self._conversation_message_content(message_id),
            metadata={
                "thinking": {
                    "status": projection.status,
                    "lines": list(projection.lines),
                }
            },
        )

    def research_progress(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._research_progress)

    def _conversation_message_content(self, message_id: str) -> str:
        with self._state_lock:
            for message in self._conversation:
                if message.message_id == message_id:
                    return message.content
        return ""

    def _on_image_progress(self, payload: dict[str, Any]) -> None:
        """Push real image job progress into the next desktop shell snapshot."""
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            return
        try:
            status = self._jarvis.runtime_service.image_status()
            job = status.get("latest_job") if isinstance(status, dict) else None
            if isinstance(job, dict) and str(job.get("job_id")) == run_id:
                with self._state_lock:
                    if not reconcile_image_job_message(self._conversation, job):
                        self._conversation.append(
                            DesktopChatMessage(
                                message_id=str(uuid4()),
                                role="assistant",
                                content=image_job_content(job),
                                metadata={
                                    "image_action": "generate",
                                    "image_job_id": run_id,
                                    "image_job_status": str(
                                        job.get("status") or "pending"
                                    ),
                                    "result": self._sanitize_dev_value(job),
                                },
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "desktop_image_progress_refresh_failed",
                extra={"exception_type": type(exc).__name__},
            )
        self._invalidate_caches()

    def is_ready(self) -> bool:
        return bool(self._jarvis.started)

    def activity_snapshot(self):
        return self._jarvis.runtime_activity_store.snapshot()

    def set_environment_status(self, env_status) -> None:
        self._env_status = env_status

    def start_backend(self) -> None:
        if self._jarvis.started:
            return
        with self._request_lock:
            if self._jarvis.started:
                return
            started_at = perf_counter()
            self._set_busy_state(True, "STARTING")
            try:
                self._jarvis.start()
                startup_ms = (perf_counter() - started_at) * 1000
                with self._state_lock:
                    self._performance["startup_ms"] = round(startup_ms, 2)
                self._logger.info(
                    "desktop_startup_completed", extra={"startup_ms": startup_ms}
                )
            finally:
                self._invalidate_caches()
                self._set_busy_state(False, "IDLE")

    def start_backend_async(self) -> Future:
        return self._executor.submit(self.start_backend)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def shell_state(self, *, force: bool = False) -> DesktopShellState:
        started_at = perf_counter()
        if not self._jarvis.started:
            state = self._loading_shell_state()
        else:
            snapshot = self._jarvis.runtime_service.snapshot(include_history=True)
            dev_runtime = self._dev_runtime_state()
            image_runtime = (
                dev_runtime.get("image_runtime")
                if isinstance(dev_runtime.get("image_runtime"), dict)
                else {}
            )
            latest_image_job = (
                image_runtime.get("latest_job")
                if isinstance(image_runtime.get("latest_job"), dict)
                else {}
            )
            if latest_image_job:
                with self._state_lock:
                    reconcile_image_job_message(self._conversation, latest_image_job)
            conversation = self._conversation_snapshot()
            llm_status = self.llm_status()
            state = DesktopShellState(
                app_name=snapshot.app_name,
                environment=snapshot.environment,
                busy=self._busy,
                activity_label=self._activity_label if self._busy else "IDLE",
                performance=dict(self._performance),
                quick_actions=list(self._quick_actions),
                panel_snapshot=self._panel_snapshot(force=force),
                conversation=conversation,
                voice=self._voice_status(force=force),
                llm_mode=llm_status["selected_mode"],
                llm_provider=llm_status["active_provider"] or "none",
                llm_model=llm_status["active_model"],
                llm_status=llm_status,
                pending_attachments=self.pending_attachments(),
                dev_runtime=dev_runtime,
            )
        elapsed_ms = (perf_counter() - started_at) * 1000
        with self._state_lock:
            self._performance["last_shell_state_ms"] = round(elapsed_ms, 2)
            state.performance = dict(self._performance)
        return state

    def llm_status(self) -> dict[str, Any]:
        selection = self._jarvis.model_service.provider_selection()
        runtime = self._jarvis.local_model_runtime_manager.status().model_dump(
            mode="json"
        )
        return {
            "selected_mode": selection.selected_mode.value,
            "active_provider": selection.active_provider,
            "request_provider": selection.request_provider,
            "active_model": selection.active_model,
            "provider_status": selection.provider_status.model_dump(mode="json"),
            "fallback_reason": selection.fallback_reason,
            "persistence_warning": selection.persistence_warning,
            "capabilities": selection.capabilities.model_dump(mode="json"),
            "key_status": self._jarvis.nvidia_secret_manager.status().model_dump(
                mode="json"
            ),
            "reasoning_preference": self.reasoning_preference().value,
            "local_runtime": runtime,
            "profile_id": runtime.get("profile_id"),
            "profile_display_name": runtime.get("display_name"),
            "profile_model_alias": runtime.get("model"),
            "profile_context_size": runtime.get("context_size"),
        }

    def reasoning_preference(self) -> ReasoningPreference:
        configured = getattr(
            self._jarvis.settings, "reasoning_preference", ReasoningPreference.AUTO
        )
        try:
            persisted = load_persistent_config(
                getattr(self._jarvis.settings, "workspace_root", None)
            ).reasoning_preference
            configured = persisted or configured
        except Exception:  # noqa: BLE001
            pass
        try:
            return ReasoningPreference(str(configured).casefold())
        except ValueError:
            return ReasoningPreference.AUTO

    def set_reasoning_preference(self, preference: str) -> dict[str, Any]:
        selected = ReasoningPreference(str(preference).casefold())
        self._jarvis.settings.reasoning_preference = selected
        project_root = getattr(self._jarvis.settings, "workspace_root", None)
        config = load_persistent_config(project_root)
        config.reasoning_preference = selected.value
        save_persistent_config(config, project_root)
        return self.llm_status()

    def set_llm_mode(self, mode: str) -> dict[str, Any]:
        self._jarvis.model_service.set_llm_mode(mode)
        return self.llm_status()

    def save_nvidia_api_key(self, value: str) -> dict[str, Any]:
        return self._jarvis.nvidia_secret_manager.save(value).model_dump(mode="json")

    def delete_nvidia_api_key(self) -> dict[str, Any]:
        return self._jarvis.nvidia_secret_manager.delete().model_dump(mode="json")

    def test_nvidia_connection(self) -> dict[str, Any]:
        provider = self._jarvis.provider_registry.get("nvidia_nim")
        return provider.status_check().model_dump(mode="json")

    def local_runtime_status(self) -> dict[str, Any]:
        diagnostic = self._jarvis.local_hardware_diagnostics.inspect(
            self._jarvis.settings.local_llm_base_url
        )
        profile = LocalLaunchProfile.from_settings(self._jarvis.settings)
        command: tuple[str, ...] | None = None
        launch_error: str | None = None
        try:
            command = self._jarvis.local_runtime_service.build_launch_command(
                "vllm",
                profile,
                diagnostic,
            )
        except LocalRuntimeIncompatibleError as exc:
            launch_error = str(exc)
        return {
            "runtime": self._jarvis.local_model_runtime_manager.status().model_dump(
                mode="json"
            ),
            "diagnostic": diagnostic.model_dump(mode="json"),
            "profile": profile.model_dump(mode="json"),
            "launch_command": list(command) if command is not None else None,
            "launch_error": launch_error,
        }

    def add_attachment_paths(self, paths: list[str | Path]) -> list[dict[str, Any]]:
        return [
            self._jarvis.attachment_service.add_path(
                self._attachment_request_id, path
            ).model_dump(mode="json")
            for path in paths
        ]

    def add_attachment_bytes(
        self,
        name: str,
        data: bytes,
        claimed_mime: str | None = None,
    ) -> dict[str, Any]:
        return self._jarvis.attachment_service.add_bytes(
            self._attachment_request_id,
            name,
            data,
            claimed_mime,
        ).model_dump(mode="json")

    def pending_attachments(self) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self._jarvis.attachment_service.list_for_request(
                self._attachment_request_id
            )
        ]

    def cancel_attachment(self, attachment_id: str) -> dict[str, Any]:
        return self._jarvis.attachment_service.cancel(
            attachment_id,
            request_id=self._attachment_request_id,
        ).model_dump(mode="json")

    def retry_attachment(self, attachment_id: str) -> dict[str, Any]:
        return self._jarvis.attachment_service.retry(
            attachment_id,
            request_id=self._attachment_request_id,
        ).model_dump(mode="json")

    def remove_attachment(self, attachment_id: str) -> None:
        self._jarvis.attachment_service.remove(
            attachment_id,
            request_id=self._attachment_request_id,
        )

    def _prepare_model_request_attachments(self, request, attachment_ids: list[str]):
        requested_ids = list(
            dict.fromkeys(str(item) for item in attachment_ids if str(item))
        )
        if not requested_ids:
            return request
        descriptors_by_id = {
            item.id: item
            for item in self._jarvis.attachment_service.list_for_request(
                self._attachment_request_id
            )
        }
        missing = [
            attachment_id
            for attachment_id in requested_ids
            if attachment_id not in descriptors_by_id
        ]
        if missing:
            raise ValueError(
                "Uno o más adjuntos ya no pertenecen a la solicitud actual."
            )
        descriptors = [
            descriptors_by_id[attachment_id] for attachment_id in requested_ids
        ]
        mode = ProcessingMode(
            self._jarvis.model_service.provider_selection().selected_mode.value
        )
        capabilities = self._attachment_provider_capabilities(mode)
        prepared = self._jarvis.multimodal_pipeline.prepare(
            descriptors,
            request_id=self._attachment_request_id,
            mode=mode,
            capabilities=capabilities,
        )
        return enrich_model_request(
            request,
            prepared,
            attachment_service=self._jarvis.attachment_service,
            request_id=self._attachment_request_id,
            context_char_budget=(
                self._jarvis.settings.conversation_context_attachment_char_budget
            ),
        )

    def _attachment_request_transform(self, attachment_ids: list[str]):
        def transform(request):
            return self._prepare_model_request_attachments(request, attachment_ids)

        return transform

    def _attachment_provider_capabilities(self, mode: ProcessingMode):
        local_provider = self._jarvis.provider_registry.get("qwen_local")
        nvidia_provider = self._jarvis.provider_registry.get("nvidia_nim")
        if mode is ProcessingMode.LOCAL:
            return local_provider.capabilities
        if mode is ProcessingMode.NVIDIA_API:
            return nvidia_provider.capabilities
        cloud_status = nvidia_provider.status_check()
        return (
            nvidia_provider.capabilities
            if cloud_status.healthy
            else local_provider.capabilities
        )

    def _remove_completed_attachments(self, attachment_ids: list[str]) -> None:
        for attachment_id in dict.fromkeys(attachment_ids):
            try:
                self._jarvis.attachment_service.remove(
                    attachment_id,
                    request_id=self._attachment_request_id,
                )
            except (KeyError, ValueError, OSError):
                continue
        if not self._jarvis.attachment_service.list_for_request(
            self._attachment_request_id
        ):
            self._attachment_request_id = f"desktop-draft-{uuid4()}"

    def refresh(self) -> DesktopShellState:
        return self.shell_state()

    def _model_history_snapshot(self) -> list[ModelMessage]:
        with self._state_lock:
            conversation = tuple(self._conversation)
            private_context = dict(self._model_context_by_message_id)
        history: list[ModelMessage] = []
        for message in conversation:
            if message.role not in {"user", "assistant"}:
                continue
            if message.role == "assistant" and message.metadata.get("streaming"):
                status = str(message.metadata.get("status") or "")
                if status not in {"completed", "cancelled", "error"}:
                    continue
            private_messages = private_context.get(message.message_id)
            model_messages = (
                [item.model_copy(deep=True) for item in private_messages]
                if private_messages is not None
                else [ModelMessage(role=message.role, content=message.content)]
            )
            for context_index, model_message in enumerate(model_messages):
                metadata = dict(model_message.metadata)
                metadata["message_id"] = (
                    message.message_id
                    if len(model_messages) == 1
                    else f"{message.message_id}:context:{context_index}"
                )
                history.append(model_message.model_copy(update={"metadata": metadata}))
        return history

    def _remember_finalized_user_context(
        self,
        message_id: str,
        request: ModelRequest,
    ) -> None:
        current = next(
            (
                message
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            None,
        )
        if current is None:
            return
        metadata = dict(current.metadata)
        metadata["message_id"] = message_id
        with self._state_lock:
            self._model_context_by_message_id[message_id] = [
                current.model_copy(
                    update={"metadata": metadata},
                    deep=True,
                )
            ]
            live_ids = {message.message_id for message in self._conversation}
            stale_ids = [
                stored_id
                for stored_id in self._model_context_by_message_id
                if stored_id not in live_ids
            ]
            for stored_id in stale_ids:
                self._model_context_by_message_id.pop(stored_id, None)

    def _remember_assistant_context(
        self,
        message_id: str,
        messages: list[ModelMessage],
    ) -> None:
        if not messages:
            return
        with self._state_lock:
            self._model_context_by_message_id[message_id] = [
                message.model_copy(deep=True) for message in messages
            ]

    def _memory_candidates_for(
        self,
        text: str,
        *,
        correlation_id: str | None,
    ):
        settings = self._jarvis.settings
        if (
            not settings.embeddings_enabled
            or settings.conversation_context_memory_top_k <= 0
        ):
            return []
        try:
            context = self._jarvis.semantic_memory_service.retrieve_context(
                SemanticSearchQuery(
                    query=text,
                    top_k=settings.conversation_context_memory_top_k,
                    correlation_id=correlation_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "conversation_memory_retrieval_failed",
                extra={
                    "correlation_id": correlation_id,
                    "error_type": type(exc).__name__,
                },
            )
            return []
        return context_fragments_from_retrieved_context(
            context,
            limit=settings.conversation_context_memory_top_k,
        )

    def send_chat(
        self,
        text: str,
        *,
        source: str = "text",
        correlation_id: str | None = None,
        metadata: dict | None = None,
        _research_cancel_event: Event | None = None,
    ):
        self.start_backend()
        if correlation_id is not None:
            with self._state_lock:
                cached = self._completed_chat_requests.get(correlation_id)
            if cached is not None:
                self._logger.info(
                    "desktop_chat_deduplicated_completed",
                    extra={"correlation_id": correlation_id, "source": source},
                )
                return cached
        queued_at = perf_counter()
        history = self._model_history_snapshot()
        memory_candidates = self._memory_candidates_for(
            text,
            correlation_id=correlation_id,
        )
        user_message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="user",
            content=text,
            metadata={
                "source": source,
                "correlation_id": correlation_id,
                **(metadata or {}),
            },
        )
        with self._state_lock:
            self._conversation.append(user_message)
            self._performance["last_first_feedback_ms"] = round(
                (perf_counter() - queued_at) * 1000, 2
            )
        self._set_busy_state(True, "PROCESSING")
        self._invalidate_caches()
        started_at = perf_counter()
        attachment_ids = list((metadata or {}).get("attachment_ids") or [])
        request_transform = None
        if attachment_ids:
            request_transform = self._attachment_request_transform(attachment_ids)

        def request_finalized(request: ModelRequest) -> None:
            self._remember_finalized_user_context(user_message.message_id, request)

        with self._request_lock:
            if request_transform is None:
                response = self._chat.handle(
                    text,
                    history=history,
                    memory_candidates=memory_candidates,
                    correlation_id=correlation_id,
                    request_finalized=request_finalized,
                    cancellation_check=(
                        _research_cancel_event.is_set
                        if _research_cancel_event is not None
                        else None
                    ),
                    search_mode=(metadata or {}).get("search_mode", "auto"),
                )
            else:
                response = self._chat.handle(
                    text,
                    request_transform=request_transform,
                    history=history,
                    memory_candidates=memory_candidates,
                    correlation_id=correlation_id,
                    request_finalized=request_finalized,
                    cancellation_check=(
                        _research_cancel_event.is_set
                        if _research_cancel_event is not None
                        else None
                    ),
                    search_mode=(metadata or {}).get("search_mode", "auto"),
                )
        elapsed_ms = (perf_counter() - started_at) * 1000
        response.message.metadata.setdefault("source", source)
        if correlation_id is not None:
            response.message.metadata.setdefault("correlation_id", correlation_id)
        with self._state_lock:
            self._conversation.append(response.message)
            self._performance["last_request_ms"] = round(elapsed_ms, 2)
            self._performance["model_latency_ms"] = self._extract_model_latency_ms(
                response.raw_result
            )
            self._performance["requests_completed"] = (
                int(self._performance["requests_completed"] or 0) + 1
            )
            self._panel_snapshot_cache = response.panel_snapshot or None
            self._panel_snapshot_cached_at = (
                perf_counter() if self._panel_snapshot_cache is not None else 0.0
            )
            if correlation_id is not None:
                self._completed_chat_requests[correlation_id] = response
                while len(self._completed_chat_requests) > 32:
                    oldest = next(iter(self._completed_chat_requests))
                    self._completed_chat_requests.pop(oldest, None)
        self._remember_assistant_context(
            response.message.message_id,
            response.context_messages,
        )
        self._logger.info(
            "desktop_chat_completed",
            extra={
                "source": source,
                "correlation_id": correlation_id,
                "request_ms": elapsed_ms,
                "first_feedback_ms": self._performance["last_first_feedback_ms"],
            },
        )
        self._set_busy_state(False, "IDLE")
        if response.message.role == "assistant":
            if response.spoken_mode == "literal":
                self._voice.speak_literal(
                    response.spoken_content or response.message.content
                )
            else:
                self._voice.speak_response(
                    response.spoken_content or response.message.content
                )
        if attachment_ids and response.raw_result.get("ok", True) is not False:
            self._remove_completed_attachments(attachment_ids)
        return response

    def send_chat_async(
        self,
        text: str,
        *,
        source: str = "text",
        correlation_id: str | None = None,
        metadata: dict | None = None,
    ) -> Future:
        request_id = correlation_id or f"desktop-chat-{uuid4().hex[:12]}"
        with self._state_lock:
            completed = self._completed_chat_requests.get(request_id)
            if completed is not None:
                future = Future()
                future.set_result(completed)
                return future
            existing = self._request_futures.get(request_id)
            if existing is not None and not existing.done():
                self._logger.info(
                    "desktop_chat_deduplicated_inflight",
                    extra={"correlation_id": request_id, "source": source},
                )
                return existing
        requested_search_mode = (metadata or {}).get("search_mode", "auto")
        resolved_search_mode = decide_search_mode(text, requested_search_mode)
        use_stream = bool((metadata or {}).get("stream")) and resolved_search_mode is SearchMode.NO_WEB
        stream_cancel_event: Event | None = None
        research_cancel_event: Event | None = None
        future: Future = Future()
        with self._state_lock:
            self._request_futures[request_id] = future
            if use_stream:
                if self._active_stream_cancel is not None:
                    self._active_stream_cancel.set()
                stream_cancel_event = Event()
                self._active_stream_id = request_id
                self._active_stream_cancel = stream_cancel_event
            elif resolved_search_mode is not SearchMode.NO_WEB:
                if self._active_research_cancel is not None:
                    self._active_research_cancel.set()
                research_cancel_event = Event()
                self._active_research_cancel = research_cancel_event
        self._set_busy_state(True, "STREAMING" if use_stream else "PROCESSING")

        def _cleanup(*, key: str = request_id) -> None:
            with self._state_lock:
                current = self._request_futures.get(key)
                if current is future:
                    self._request_futures.pop(key, None)
                if (
                    research_cancel_event is not None
                    and self._active_research_cancel is research_cancel_event
                ):
                    self._active_research_cancel = None

        def _copy_result(worker_future: Future) -> None:
            if future.cancelled():
                _cleanup()
                return
            try:
                future.set_result(worker_future.result())
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)
            finally:
                _cleanup()

        def _start_worker() -> None:
            if future.cancelled():
                _cleanup()
                return
            if use_stream:
                worker_future = self._executor.submit(
                    self.send_chat_streaming,
                    text,
                    source=source,
                    correlation_id=request_id,
                    metadata=metadata,
                    _admitted_cancel_event=stream_cancel_event,
                )
            else:
                worker_future = self._executor.submit(
                    self.send_chat,
                    text,
                    source=source,
                    correlation_id=request_id,
                    metadata=metadata,
                    _research_cancel_event=research_cancel_event,
                )
            worker_future.add_done_callback(_copy_result)

        Timer(0.01, _start_worker).start()
        return future

    def cancel_active_research(self) -> bool:
        with self._state_lock:
            event = self._active_research_cancel
            if event is None:
                return False
            event.set()
            return True

    def send_chat_streaming(
        self,
        text: str,
        *,
        source: str = "text",
        correlation_id: str | None = None,
        metadata: dict | None = None,
        _admitted_cancel_event: Event | None = None,
    ):
        self.start_backend()
        resolved_search_mode = decide_search_mode(
            text,
            (metadata or {}).get("search_mode", "auto"),
        )
        if resolved_search_mode is not SearchMode.NO_WEB:
            return self.send_chat(
                text,
                source=source,
                correlation_id=correlation_id,
                metadata={**(metadata or {}), "stream": False},
                _research_cancel_event=_admitted_cancel_event,
            )
        correlation_id = correlation_id or f"desktop-stream-{uuid4().hex[:12]}"
        route_to_agent = self._should_route_stream_to_agent(text)
        history = self._model_history_snapshot()
        memory_candidates = self._memory_candidates_for(
            text,
            correlation_id=correlation_id,
        )
        cancel_event = _admitted_cancel_event or Event()
        if _admitted_cancel_event is None:
            with self._state_lock:
                if self._active_stream_cancel is not None:
                    self._active_stream_cancel.set()
                self._active_stream_id = correlation_id
                self._active_stream_cancel = cancel_event
        queued_at = perf_counter()
        user_message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="user",
            content=text,
            metadata={
                "source": source,
                "correlation_id": correlation_id,
                **(metadata or {}),
            },
        )
        assistant_message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="assistant",
            # Provider selection is activity/status metadata, never assistant
            # conversation content.  In LOCAL this stale placeholder used to
            # Do not expose provider identity before the first model delta.
            content="",
            metadata={
                "source": source,
                "correlation_id": correlation_id,
                "streaming": True,
                "status": "thinking",
                "thinking": {
                    "status": "active",
                    "lines": ["Analizando la solicitud…"],
                },
            },
        )
        with self._state_lock:
            self._conversation.append(user_message)
            self._conversation.append(assistant_message)
            self._progress_message_by_correlation[correlation_id] = (
                assistant_message.message_id
            )
            self._performance["last_first_feedback_ms"] = round(
                (perf_counter() - queued_at) * 1000, 2
            )
        self._set_busy_state(True, "STREAMING")
        self._invalidate_caches()
        started_at = perf_counter()
        full_text = ""
        raw_result: dict[str, Any] = {
            "status": "streaming",
            "streaming": True,
            "request_id": correlation_id,
            "stream_started": True,
            "first_chunk_received": False,
            "last_chunk_received": False,
            "chunk_count": 0,
            "stream_closed_normally": False,
            "provider_text_length": 0,
            "desktop_authoritative_text_length": 0,
            "desktop_visual_text_length": 0,
            "final_render_source_length": 0,
            "final_render_completed": False,
        }
        spoken_content: str | None = None
        attachment_ids = list((metadata or {}).get("attachment_ids") or [])
        execution_context = RequestExecutionContext(request_id=correlation_id)
        for attachment_id in attachment_ids:
            execution_context.register_file(str(attachment_id))
        try:
            with self._request_lock:
                if route_to_agent:
                    agent_started_at = perf_counter()
                    streamed_agent_text = ""

                    def agent_stream_callback(delta: str) -> None:
                        nonlocal streamed_agent_text
                        streamed_agent_text += delta
                        self._replace_conversation_message(
                            assistant_message.message_id,
                            streamed_agent_text,
                            metadata={
                                "streaming": True,
                                "status": "streaming",
                                "agent": True,
                            },
                        )

                    attachment_transform = None
                    if attachment_ids:
                        attachment_transform = self._attachment_request_transform(
                            attachment_ids
                        )

                    def request_transform(request: ModelRequest) -> ModelRequest:
                        if attachment_transform is not None:
                            request = attachment_transform(request)
                        return request.model_copy(
                            update={
                                "metadata": {
                                    **request.metadata,
                                    "agent_tool_required": True,
                                }
                            }
                        )

                    def request_finalized(request: ModelRequest) -> None:
                        self._remember_finalized_user_context(
                            user_message.message_id,
                            request,
                        )

                    response = self._chat.handle(
                        text,
                        request_transform=request_transform,
                        history=history,
                        memory_candidates=memory_candidates,
                        correlation_id=correlation_id,
                        request_finalized=request_finalized,
                        stream_callback=agent_stream_callback,
                        cancellation_check=cancel_event.is_set,
                    )
                    raw_result = response.raw_result
                    raw_result["streaming"] = True
                    raw_result["agent_routed"] = True
                    progress_snapshot = self._jarvis.runtime_activity_store.snapshot(
                        correlation_id=correlation_id
                    )
                    progress_projection = project_runtime_progress(progress_snapshot)
                    raw_result["agent_progress_events"] = [
                        {
                            "event_name": event.event_name,
                            "state": event.state.value,
                            "action": event.action,
                            "call_id": event.call_id,
                        }
                        for event in progress_snapshot.events
                    ]
                    raw_result["agent_execution_summary"] = (
                        progress_projection.safe_summary
                    )
                    raw_result["safe_reasoning_summary"] = (
                        progress_projection.safe_summary
                    )
                    raw_result["reasoning_duration_seconds"] = round(
                        perf_counter() - agent_started_at,
                        3,
                    )
                    raw_result["thinking_lines"] = list(progress_projection.lines)
                    raw_result["thinking_status"] = progress_projection.status
                    authoritative_text = str(
                        (
                            raw_result.get("final_answer")
                            if raw_result.get("status") == "completed"
                            else response.message.content
                        )
                        or ""
                    )
                    raw_result.setdefault(
                        "provider_text_length", len(streamed_agent_text)
                    )
                    raw_result["desktop_authoritative_text_length"] = len(
                        authoritative_text
                    )
                    raw_result["desktop_visual_text_length"] = len(
                        authoritative_text
                    )
                    raw_result["final_render_source_length"] = len(
                        authoritative_text
                    )
                    raw_result["final_render_completed"] = (
                        raw_result.get("status") == "completed"
                        and raw_result.get("stream_closed_normally") is not False
                    )
                    spoken_content = response.spoken_content or response.message.content
                    self._remember_assistant_context(
                        assistant_message.message_id,
                        response.context_messages,
                    )
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        response.message.content,
                        metadata={
                            "streaming": False,
                            "status": raw_result.get("status", "completed"),
                            "agent": True,
                            "result": raw_result,
                            "thinking": {
                                "status": progress_projection.status,
                                "lines": list(progress_projection.lines),
                            },
                            "thought": {
                                "status": (
                                    "completed"
                                    if raw_result.get("status")
                                    in {"completed", "truncated"}
                                    else "failed"
                                ),
                                "elapsed_seconds": raw_result.get(
                                    "reasoning_duration_seconds"
                                ),
                                "summary": progress_projection.safe_summary,
                            },
                        },
                    )
                    return self._stream_response(
                        assistant_message,
                        response.message.content,
                        raw_result,
                    )
                prepared = self._chat.prepare_streaming_request(
                    text.strip(),
                    history=history,
                )
                if prepared is None:
                    request_transform = None
                    if attachment_ids:
                        request_transform = self._attachment_request_transform(
                            attachment_ids
                        )

                    def request_finalized(request: ModelRequest) -> None:
                        self._remember_finalized_user_context(
                            user_message.message_id,
                            request,
                        )

                    if request_transform is None:
                        response = self._chat.handle(
                            text,
                            history=history,
                            memory_candidates=memory_candidates,
                            correlation_id=correlation_id,
                            request_finalized=request_finalized,
                        )
                    else:
                        response = self._chat.handle(
                            text,
                            request_transform=request_transform,
                            history=history,
                            memory_candidates=memory_candidates,
                            correlation_id=correlation_id,
                            request_finalized=request_finalized,
                        )
                    raw_result = response.raw_result
                    spoken_content = response.spoken_content or response.message.content
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        response.message.content,
                        metadata={
                            "streaming": False,
                            "status": "completed",
                            "result": raw_result,
                        },
                    )
                    return response
                if hasattr(prepared, "message"):
                    response = prepared
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        response.message.content,
                        metadata={
                            "streaming": False,
                            "status": "completed",
                            "result": response.raw_result,
                        },
                    )
                    raw_result = response.raw_result
                    spoken_content = response.spoken_content or response.message.content
                    return response
                request, web_result = prepared
                if attachment_ids:
                    request = self._prepare_model_request_attachments(
                        request, attachment_ids
                    )
                    raw_result["attachments"] = list(
                        request.metadata.get("attachment_routes") or []
                    )
                if web_result is not None:
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        f"Encontré {len(web_result.hits)} fuentes. Redactando con el proveedor asignado...",
                        metadata={
                            "streaming": True,
                            "status": "synthesizing",
                            "web_search": web_result.model_dump(mode="json"),
                        },
                    )
                    raw_result["web_search"] = web_result.model_dump(mode="json")
                request = request.model_copy(update={"correlation_id": correlation_id})
                request = self._chat.finalize_model_request(
                    request,
                    history=history,
                    memory_candidates=memory_candidates,
                )
                self._remember_finalized_user_context(
                    user_message.message_id,
                    request,
                )
                chunks = 0
                terminal_seen = False
                reasoning_seen = False
                reasoning_started_at: float | None = None
                reasoning_finished_at: float | None = None
                first_token_ms: float | None = None
                for chunk in self._jarvis.runtime_service.stream_model(
                    request,
                    cancel_check=cancel_event.is_set,
                    execution_context=execution_context,
                ):
                    if (
                        not self._is_active_stream(correlation_id)
                        or cancel_event.is_set()
                    ):
                        raw_result["status"] = "cancelled"
                        return self._stream_response(
                            assistant_message, "", raw_result, cancelled=True
                        )
                    if chunk.error:
                        failure = classify_provider_failure(chunk.error)
                        message = provider_failure_message(
                            failure,
                            selected_mode=self.llm_status()["selected_mode"],
                            partial_output=full_text or None,
                        )
                        raw_result.update(
                            {
                                "status": "error",
                                "error": failure.detail,
                                "error_code": failure.code,
                                "exception_category": type(chunk.error).__name__,
                            }
                        )
                        self._replace_conversation_message(
                            assistant_message.message_id,
                            message,
                            metadata={
                                "streaming": False,
                                "status": "error",
                                "result": raw_result,
                            },
                        )
                        return self._stream_response(
                            assistant_message, message, raw_result
                        )
                    if chunk.metadata.get("reasoning_active") and not reasoning_seen:
                        reasoning_seen = True
                        reasoning_started_at = perf_counter()
                        raw_result["reasoning_detected"] = True
                        self._replace_conversation_message(
                            assistant_message.message_id,
                            "Pensando…",
                            metadata={
                                "streaming": True,
                                "status": "thinking",
                                "reasoning_active": True,
                                "thought": {"status": "active"},
                            },
                        )
                    if chunk.text:
                        if reasoning_seen and reasoning_finished_at is None:
                            reasoning_finished_at = perf_counter()
                        raw_result["first_chunk_received"] = True
                        raw_result["last_chunk_received"] = True
                        if first_token_ms is None:
                            first_token_ms = round(
                                (perf_counter() - started_at) * 1000, 2
                            )
                        chunks += 1
                        raw_result["chunk_count"] = chunks
                        full_text += chunk.text
                        raw_result["provider_text_length"] = len(full_text)
                        raw_result["desktop_authoritative_text_length"] = len(full_text)
                        raw_result["desktop_visual_text_length"] = len(full_text)
                        self._replace_conversation_message(
                            assistant_message.message_id,
                            full_text,
                            metadata={
                                "streaming": True,
                                "status": "streaming",
                                "chunks": chunks,
                                "thinking": {
                                    "status": "active",
                                    "lines": [
                                        "Analizando la solicitud…",
                                        "Preparando respuesta…",
                                    ],
                                },
                                **(
                                    {"thought": {"status": "active"}}
                                    if reasoning_seen
                                    else {}
                                ),
                            },
                        )
                    if chunk.done:
                        terminal_seen = True
                        if reasoning_seen and reasoning_finished_at is None:
                            reasoning_finished_at = perf_counter()
                        finish_reason = chunk.metadata.get("finish_reason")
                        stream_closed_normally = bool(
                            chunk.metadata.get(
                                "stream_closed_normally",
                                not chunk.metadata.get("incomplete", False),
                            )
                        )
                        completion_accepted = (
                            stream_closed_normally
                            and finish_reason == "stop"
                            and not chunk.metadata.get("incomplete", False)
                        )
                        truncated = (
                            stream_closed_normally and finish_reason == "length"
                        )
                        raw_result.update(
                            {
                                "status": (
                                    "completed"
                                    if completion_accepted
                                    else "truncated"
                                    if truncated
                                    else "interrupted"
                                ),
                                "outcome_code": (
                                    "NORMAL_STOP"
                                    if completion_accepted
                                    else "MODEL_OUTPUT_TRUNCATED"
                                    if truncated
                                    else "PROVIDER_TRANSPORT_FAILURE"
                                ),
                                "provider_name": chunk.metadata.get("provider")
                                or chunk.metadata.get("active_provider"),
                                "model_name": chunk.metadata.get("model")
                                or chunk.metadata.get("active_model"),
                                "content": full_text,
                                "chunks": chunks,
                                "first_token_ms": first_token_ms,
                                "latency_ms": round(
                                    (perf_counter() - started_at) * 1000, 2
                                ),
                                "reasoning_detected": reasoning_seen
                                or bool(chunk.metadata.get("reasoning_chunks")),
                                "finish_reason": finish_reason,
                                "completion_tokens": chunk.metadata.get("completion_tokens"),
                                "prompt_tokens": chunk.metadata.get("prompt_tokens"),
                                "auto_continuation_count": int(
                                    chunk.metadata.get("auto_continuation_count") or 0
                                ),
                                "model_request_count": chunk.metadata.get(
                                    "model_request_count", 1
                                ),
                                "segment_finish_reasons": list(
                                    chunk.metadata.get("segment_finish_reasons") or []
                                ),
                                "segment_durations_seconds": list(
                                    chunk.metadata.get("segment_durations_seconds") or []
                                ),
                                "segment_completion_tokens": list(
                                    chunk.metadata.get("segment_completion_tokens") or []
                                ),
                                "segment_visible_tokens_estimated": list(
                                    chunk.metadata.get(
                                        "segment_visible_tokens_estimated"
                                    )
                                    or []
                                ),
                                "segment_reasoning_chunks": list(
                                    chunk.metadata.get("segment_reasoning_chunks")
                                    or []
                                ),
                                "segment_max_tokens": list(
                                    chunk.metadata.get("segment_max_tokens") or []
                                ),
                                "output_request_class": chunk.metadata.get(
                                    "output_request_class"
                                ),
                                "total_completion_tokens": chunk.metadata.get(
                                    "total_completion_tokens"
                                ),
                                "provider_completion_tokens": chunk.metadata.get(
                                    "provider_completion_tokens"
                                ),
                                "visible_response_tokens_estimated": (
                                    chunk.metadata.get(
                                        "visible_response_tokens_estimated"
                                    )
                                ),
                                "total_response_token_limit": chunk.metadata.get(
                                    "total_response_token_limit"
                                ),
                                "visible_response_soft_target": chunk.metadata.get(
                                    "visible_response_soft_target"
                                ),
                                "visible_response_hard_ceiling": chunk.metadata.get(
                                    "visible_response_hard_ceiling"
                                ),
                                "logical_response_remaining": chunk.metadata.get(
                                    "logical_response_remaining"
                                ),
                                "rolling_context_tokens": list(
                                    chunk.metadata.get("rolling_context_tokens") or []
                                ),
                                "rolling_continuation_count": int(
                                    chunk.metadata.get(
                                        "rolling_continuation_count"
                                    )
                                    or 0
                                ),
                                "loop_detection_count": int(
                                    chunk.metadata.get("loop_detection_count") or 0
                                ),
                                "loop_correction_attempted": bool(
                                    chunk.metadata.get(
                                        "loop_correction_attempted"
                                    )
                                ),
                                "deduplicated_overlap_chars": int(
                                    chunk.metadata.get("deduplicated_overlap_chars") or 0
                                ),
                                "context_prevented_continuation": bool(
                                    chunk.metadata.get(
                                        "context_prevented_continuation"
                                    )
                                ),
                                "auto_continuation_failed": bool(
                                    chunk.metadata.get("auto_continuation_failed")
                                ),
                                "continuation_stop_reason": chunk.metadata.get(
                                    "continuation_stop_reason"
                                ),
                                "original_finish_reason": chunk.metadata.get(
                                    "original_finish_reason"
                                ),
                                "stream_closed_normally": stream_closed_normally,
                                "completion_accepted": completion_accepted,
                                "exception_category": (
                                    None
                                    if completion_accepted or truncated
                                    else (
                                        "STREAM_TRANSPORT"
                                    )
                                ),
                                "last_chunk_received": True,
                            }
                        )
                        if reasoning_started_at is not None and reasoning_finished_at is not None:
                            raw_result["reasoning_duration_seconds"] = round(
                                reasoning_finished_at - reasoning_started_at, 3
                            )
                        break
                if not terminal_seen:
                    raw_result.update(
                        {
                            "status": "interrupted",
                            "outcome_code": "PROVIDER_TRANSPORT_FAILURE",
                            "finish_reason": None,
                            "stream_closed_normally": False,
                            "completion_accepted": False,
                            "exception_category": "STREAM_TRANSPORT",
                            "provider_text_length": len(full_text),
                            "desktop_authoritative_text_length": len(full_text),
                        }
                    )
            final_text = full_text
            if not final_text:
                if raw_result.get("status") == "truncated":
                    final_text = "La respuesta alcanzó el límite de salida sin contenido visible."
                elif raw_result.get("status") == "interrupted":
                    final_text = "La conexión con el proveedor se interrumpió antes de producir contenido."
                else:
                    final_text = (
                        "No obtuve una salida valida del sistema, pero sigo listo para responder."
                    )
                    raw_result["absolute_fallback_used"] = True
            if raw_result.get("status") == "truncated" and full_text:
                final_text = (
                    f"{final_text.rstrip()}\n\n"
                    "La respuesta alcanzó el límite de salida y quedó incompleta."
                )
            if raw_result.get("status") == "interrupted":
                interruption = (
                    "La generación local se interrumpió antes de finalizar."
                    if self.llm_status()["selected_mode"] == "local"
                    else "La generación se interrumpió antes de finalizar."
                )
                final_text = (
                    f"{final_text.rstrip()}\n\n{interruption}"
                )
            spoken_content = final_text
            raw_result["final_render_source_length"] = len(final_text)
            raw_result["final_render_completed"] = True
            thought_metadata = {}
            safe_summary = (
                "• Analicé la solicitud sin exponer razonamiento interno.\n"
                "• Organicé la respuesta visible alrededor de la petición."
            )
            if raw_result.get("reasoning_duration_seconds") is None:
                raw_result["reasoning_duration_seconds"] = round(
                    perf_counter() - started_at,
                    3,
                )
                raw_result["reasoning_duration_source"] = "response_preparation"
            raw_result["safe_reasoning_summary"] = safe_summary
            thought_metadata["thought"] = {
                "status": (
                    "completed"
                    if raw_result.get("status") in {"completed", "truncated"}
                    else "failed"
                ),
                "elapsed_seconds": raw_result.get("reasoning_duration_seconds"),
                "summary": safe_summary,
            }
            thinking_lines = ["Analizando la solicitud…"]
            if raw_result.get("auto_continuation_count"):
                thinking_lines.append("Continuando respuesta…")
            thinking_lines.append("Preparando respuesta…")
            thinking_metadata = {
                "thinking": {
                    "status": (
                        "completed"
                        if raw_result.get("status") == "completed"
                        else "failed"
                    ),
                    "lines": thinking_lines,
                }
            }
            raw_result["thinking_lines"] = list(
                thinking_metadata["thinking"]["lines"]
            )
            raw_result["thinking_status"] = thinking_metadata["thinking"]["status"]
            self._replace_conversation_message(
                assistant_message.message_id,
                final_text,
                metadata={
                    "streaming": False,
                    "status": raw_result.get("status", "completed"),
                    "result": raw_result,
                    **thinking_metadata,
                    **thought_metadata,
                },
            )
            return self._stream_response(assistant_message, final_text, raw_result)
        except Exception as exc:  # noqa: BLE001
            failure = classify_provider_failure(exc)
            message = provider_failure_message(
                failure,
                selected_mode=self.llm_status()["selected_mode"],
                partial_output=full_text or None,
            )
            raw_result.update(
                {
                    "status": "error",
                    "outcome_code": "PROVIDER_TRANSPORT_FAILURE",
                    "error": failure.detail,
                    "error_code": failure.code,
                    "stream_closed_normally": False,
                    "exception_category": type(exc).__name__,
                    "provider_text_length": len(full_text),
                    "desktop_authoritative_text_length": len(full_text),
                    "desktop_visual_text_length": len(full_text),
                }
            )
            self._replace_conversation_message(
                assistant_message.message_id,
                message,
                metadata={"streaming": False, "status": "error", "result": raw_result},
            )
            self._logger.warning(
                "desktop_stream_integrity",
                extra={
                    "request_id": correlation_id,
                    "status": raw_result.get("status"),
                    "provider": raw_result.get("provider_name"),
                    "model": raw_result.get("model_name"),
                    "chunk_count": raw_result.get("chunk_count", 0),
                    "provider_text_length": raw_result.get("provider_text_length", 0),
                    "desktop_authoritative_text_length": raw_result.get("desktop_authoritative_text_length", 0),
                    "final_render_source_length": raw_result.get("final_render_source_length", 0),
                    "finish_reason": raw_result.get("finish_reason"),
                    "stream_closed_normally": raw_result.get("stream_closed_normally", False),
                    "exception_category": raw_result.get("exception_category"),
                },
            )
            return self._stream_response(assistant_message, message, raw_result)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            released_active_stream = False
            with self._state_lock:
                self._progress_message_by_correlation.pop(correlation_id, None)
                self._performance["last_request_ms"] = round(elapsed_ms, 2)
                self._performance["model_latency_ms"] = raw_result.get("latency_ms")
                self._performance["requests_completed"] = (
                    int(self._performance["requests_completed"] or 0) + 1
                )
                if (
                    self._active_stream_id == correlation_id
                    and self._active_stream_cancel is cancel_event
                ):
                    self._active_stream_id = None
                    self._active_stream_cancel = None
                    released_active_stream = True
                self._completed_chat_requests[correlation_id] = self._stream_response(
                    assistant_message, spoken_content or full_text, raw_result
                )
                self._logger.info(
                    "desktop_stream_integrity",
                    extra={
                        "request_id": correlation_id,
                        "status": raw_result.get("status"),
                        "provider": raw_result.get("provider_name"),
                        "model": raw_result.get("model_name"),
                        "chunk_count": raw_result.get("chunk_count", 0),
                        "provider_text_length": raw_result.get("provider_text_length", 0),
                        "desktop_authoritative_text_length": raw_result.get("desktop_authoritative_text_length", 0),
                        "final_render_source_length": raw_result.get("final_render_source_length", 0),
                        "finish_reason": raw_result.get("finish_reason"),
                        "stream_closed_normally": raw_result.get("stream_closed_normally", False),
                        "exception_category": raw_result.get("exception_category"),
                    },
                )
            if released_active_stream:
                self._set_busy_state(False, "IDLE")
            if raw_result.get("status") == "completed" and spoken_content:
                if attachment_ids:
                    self._remove_completed_attachments(attachment_ids)
                self._voice.speak_response(spoken_content)

    def _should_route_stream_to_agent(self, text: str) -> bool:
        if self._explicit_agent_action_intent(text):
            return True
        settings = self._jarvis.settings
        router = getattr(self._jarvis, "neural_router_shadow", None)
        if router is None or not settings.neural_router_routing_enabled:
            return False
        if not self._has_operational_language(text):
            return False
        try:
            prediction = router.predict(text, top_k=3)
            if not router.promoted:
                return False
            top_score = prediction.scores[0].score if prediction.scores else 0.0
            second_score = (
                prediction.scores[1].score if len(prediction.scores) > 1 else 0.0
            )
            if top_score < settings.neural_router_min_confidence:
                return False
            if top_score - second_score < settings.neural_router_min_score_margin:
                return False
            intended_tools = set(router.tools_for_intent(prediction.intent))
            available_tools = {
                schema["function"]["name"]
                for schema in self._jarvis.action_executor.tool_schemas()
                if isinstance(schema.get("function"), dict)
                and isinstance(schema["function"].get("name"), str)
            }
            return bool(intended_tools & available_tools)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "neural_router_runtime_prediction_failed",
                extra={"error_type": type(exc).__name__},
            )
            return False

    def _explicit_agent_action_intent(self, text: str) -> bool:
        """Route narrow imperative machine/file requests, never how-to questions."""

        folded = " ".join(text.casefold().split())
        informational = folded.startswith(
            ("¿cómo", "como ", "cómo ", "¿qué", "que ", "qué ", "how ", "what ")
        ) or any(
            marker in folded
            for marker in ("como puedo", "cómo puedo", "qué comando", "que comando")
        )
        if informational:
            return False
        available = {
            schema["function"]["name"]
            for schema in self._jarvis.action_executor.tool_schemas()
            if isinstance(schema.get("function"), dict)
            and isinstance(schema["function"].get("name"), str)
        }
        dns_action = (
            any(word in folded for word in ("limpia", "borrar", "borra", "vacía", "vacia", "flush"))
            and "dns" in folded
            and any(word in folded for word in ("cache", "caché"))
            and {"terminal.start_session", "terminal.run"} <= available
        )
        file_search = (
            any(word in folded for word in ("busca", "buscar", "encuentra", "localiza"))
            and any(word in folded for word in ("archivo", "archivos", "ruta", "dirección", "direccion", "carpeta"))
            and "files.search" in available
        )
        calculation = (
            any(word in folded for word in ("calcula", "calcular", "calculate"))
            and "calculator.evaluate" in available
        )
        return dns_action or file_search or calculation

    @staticmethod
    def _has_operational_language(text: str) -> bool:
        folded = " ".join(text.casefold().split())
        if folded.startswith(("¿cómo", "como ", "cómo ", "how ", "what ")):
            return False
        return any(
            word in folded
            for word in (
                "abre ",
                "busca ",
                "calcula ",
                "crea ",
                "ejecuta ",
                "encuentra ",
                "investiga ",
                "limpia ",
                "localiza ",
                "mueve ",
                "renombra ",
                "elimina ",
                "borra ",
                "verifica ",
            )
        )

    def _replace_conversation_message(
        self, message_id: str, content: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._state_lock:
            for index, message in enumerate(self._conversation):
                if message.message_id == message_id:
                    merged = dict(message.metadata)
                    if metadata:
                        merged.update(metadata)
                    self._conversation[index] = message.model_copy(
                        update={"content": content, "metadata": merged}
                    )
                    break
        self._invalidate_caches()

    def _is_active_stream(self, correlation_id: str) -> bool:
        with self._state_lock:
            return self._active_stream_id == correlation_id

    @staticmethod
    def _stream_response(
        message: DesktopChatMessage,
        content: str,
        raw_result: dict[str, Any],
        *,
        cancelled: bool = False,
    ):
        presentation: dict[str, Any] = {}
        thinking_lines = raw_result.get("thinking_lines")
        if isinstance(thinking_lines, list):
            presentation["thinking"] = {
                "status": raw_result.get("thinking_status") or "completed",
                "lines": [str(line) for line in thinking_lines if str(line).strip()],
            }
        safe_summary = raw_result.get("safe_reasoning_summary")
        if isinstance(safe_summary, str) and safe_summary.strip():
            presentation["thought"] = {
                "status": (
                    "completed"
                    if raw_result.get("status") in {"completed", "truncated"}
                    else "failed"
                ),
                "elapsed_seconds": raw_result.get("reasoning_duration_seconds"),
                "summary": safe_summary,
            }
        final_message = message.model_copy(
            update={
                "content": content,
                "metadata": {
                    **message.metadata,
                    **presentation,
                    "result": raw_result,
                },
            }
        )
        from .base import DesktopChatResponse

        return DesktopChatResponse(
            message=final_message,
            spoken_content=None if cancelled else content,
            spoken_mode="prepared",
            raw_result=raw_result,
        )

    def execute_quick_action(
        self, action_id: str, *, payload: dict | None = None
    ) -> dict:
        self.start_backend()
        self._set_busy_state(True, f"EXECUTING {action_id.upper()}")
        started_at = perf_counter()
        with self._request_lock:
            result = self._actions.execute(action_id, payload=payload)
        elapsed_ms = (perf_counter() - started_at) * 1000
        summary = self._summarize_quick_action(action_id, result)
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={"quick_action": action_id, "result": result},
                )
            )
            self._performance["last_quick_action_ms"] = round(elapsed_ms, 2)
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        self._logger.info(
            "desktop_quick_action_completed",
            extra={"action_id": action_id, "request_ms": elapsed_ms},
        )
        return result

    def execute_quick_action_async(
        self, action_id: str, *, payload: dict | None = None
    ) -> Future:
        return self._executor.submit(
            self.execute_quick_action, action_id, payload=payload
        )

    def execute_dev_action(
        self, action_id: str, *, payload: dict | None = None
    ) -> dict:
        self.start_backend()
        payload = payload or {}
        self._set_busy_state(True, f"CODE {action_id.upper()}")
        started_at = perf_counter()
        try:
            result = self._run_dev_action(action_id, payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception(
                "desktop_dev_action_failed",
                extra={"action_id": action_id, "exception_type": type(exc).__name__},
            )
            result = {
                "status": "failed",
                "action": action_id,
                "message": str(exc),
                "error_type": type(exc).__name__,
            }
        result = self._sanitize_dev_value(result)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        if isinstance(result, dict):
            result.setdefault("action", action_id)
            result["elapsed_ms"] = elapsed_ms
        summary = self._summarize_dev_action(
            action_id, result if isinstance(result, dict) else {"result": result}
        )
        with self._state_lock:
            self._last_dev_result = (
                result
                if isinstance(result, dict)
                else {"status": "ok", "action": action_id, "result": result}
            )
            self._last_dev_result.setdefault("summary", summary)
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={"dev_action": action_id, "result": self._last_dev_result},
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        self._logger.info(
            "desktop_dev_action_completed",
            extra={"action_id": action_id, "request_ms": elapsed_ms},
        )
        return self._last_dev_result

    def execute_dev_action_async(
        self, action_id: str, *, payload: dict | None = None
    ) -> Future:
        return self._executor.submit(
            self.execute_dev_action, action_id, payload=payload
        )

    def confirm_latest_agent_action(self) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "AGENT CONFIRM")
        started_at = perf_counter()
        try:
            mission_id = self._latest_desktop_agent_mission_id()
            receipt = self._jarvis.runtime_service.desktop_agent_confirm(mission_id)
            result = receipt.model_dump(mode="json")
            summary = str(
                result.get("summary") or "Accion confirmada y ejecutada por Agent Mode."
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_agent_confirm_failed",
                extra={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            result = {
                "status": "failed",
                "message": self._sanitize_dev_value(str(exc)),
                "error_type": type(exc).__name__,
            }
            summary = f"No pude confirmar la accion del agente: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={
                        "agent_action": "confirm",
                        "result": self._sanitize_dev_value(result),
                    },
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def confirm_latest_agent_action_async(self) -> Future:
        return self._executor.submit(self.confirm_latest_agent_action)

    def dry_run_agent_action(self, text: str | None = None) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "AGENT DRY RUN")
        started_at = perf_counter()
        goal = (text or "").strip()
        if not goal:
            with self._state_lock:
                for message in reversed(self._conversation):
                    if message.role == "user" and str(message.content or "").strip():
                        goal = str(message.content).strip()
                        break
        try:
            if not goal:
                raise RuntimeError("escribe un objetivo antes de simular")
            result = self._jarvis.runtime_service.desktop_agent_dry_run(
                {
                    "goal": goal,
                    "metadata": {"source": "desktop_ui", "dry_run": True},
                    "source_surface": "desktop_chat",
                }
            )
            summary = str(
                result.get("summary") or "Dry run completado sin ejecutar acciones."
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_agent_dry_run_failed",
                extra={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            result = {
                "status": "failed",
                "message": self._sanitize_dev_value(str(exc)),
                "error_type": type(exc).__name__,
            }
            summary = f"No pude simular la accion del agente: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={
                        "agent_action": "dry_run",
                        "result": self._sanitize_dev_value(result),
                    },
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def dry_run_agent_action_async(self, text: str | None = None) -> Future:
        return self._executor.submit(self.dry_run_agent_action, text)

    def stop_latest_agent(self) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "AGENT STOP")
        started_at = perf_counter()
        try:
            mission_id = self._latest_desktop_agent_mission_id()
            receipt = self._jarvis.runtime_service.desktop_agent_abort(
                mission_id, reason="stop agent requested from desktop UI"
            )
            result = receipt.model_dump(mode="json")
            summary = str(result.get("summary") or "Agent Mode detenido.")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_agent_stop_failed",
                extra={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            result = {
                "status": "failed",
                "message": self._sanitize_dev_value(str(exc)),
                "error_type": type(exc).__name__,
            }
            summary = f"No pude detener el agente: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={
                        "agent_action": "stop",
                        "result": self._sanitize_dev_value(result),
                    },
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def stop_latest_agent_async(self) -> Future:
        return self._executor.submit(self.stop_latest_agent)

    def generate_image(self, prompt: str) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "IMAGE GENERATION")
        started_at = perf_counter()
        try:
            from jarvis.image_runtime.prompting import build_image_request_from_text

            built = build_image_request_from_text(
                prompt, defaults=self._jarvis.settings
            )
            if not built.allowed or built.request is None:
                raise RuntimeError(built.reason)
            job = self._jarvis.runtime_service.image_generate(built.request)
            result = job.model_dump(mode="json")
            summary = image_job_content(result)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_image_generate_failed",
                extra={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            result = {
                "status": "failed",
                "message": self._sanitize_dev_value(str(exc)),
                "error_type": type(exc).__name__,
            }
            summary = f"No pude iniciar la generacion de imagen: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            if not reconcile_image_job_message(self._conversation, result):
                self._conversation.append(
                    DesktopChatMessage(
                        message_id=str(uuid4()),
                        role="assistant",
                        content=summary,
                        metadata={
                            "image_action": "generate",
                            "image_job_id": result.get("job_id"),
                            "image_job_status": result.get("status"),
                            "result": self._sanitize_dev_value(result),
                        },
                    )
                )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def generate_image_async(self, prompt: str) -> Future:
        return self._executor.submit(self.generate_image, prompt)

    def cancel_image_generation(self) -> dict[str, Any]:
        self.start_backend()
        result = self._jarvis.runtime_service.image_cancel()
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=str(result.get("message") or "Cancelacion solicitada."),
                    metadata={
                        "image_action": "cancel",
                        "result": self._sanitize_dev_value(result),
                    },
                )
            )
        self._invalidate_caches()
        return result

    def cancel_image_generation_async(self) -> Future:
        return self._executor.submit(self.cancel_image_generation)

    def unload_image_model(self) -> dict[str, Any]:
        self.start_backend()
        result = self._jarvis.runtime_service.image_unload()
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=str(
                        result.get("message")
                        or "Modelo de imagen descargado de memoria."
                    ),
                    metadata={
                        "image_action": "unload",
                        "result": self._sanitize_dev_value(result),
                    },
                )
            )
        self._invalidate_caches()
        return result

    def unload_image_model_async(self) -> Future:
        return self._executor.submit(self.unload_image_model)

    def open_image_output_folder_async(self) -> Future:
        def _open() -> dict[str, Any]:
            self.start_backend()
            status = self._jarvis.runtime_service.image_status()
            folder = str(
                status.get("output_dir")
                or self._jarvis.settings.resolved_image_output_dir
            )
            receipt = self._jarvis.runtime_service.system_open(
                {
                    "query": folder,
                    "metadata": {"source": "desktop_image_studio", "approved": True},
                }
            )
            return (
                receipt.model_dump(mode="json")
                if hasattr(receipt, "model_dump")
                else dict(receipt)
            )

        return self._executor.submit(_open)

    def _latest_desktop_agent_mission_id(self) -> str:
        missions = self._jarvis.runtime_service.desktop_agent_list()
        if not missions:
            raise RuntimeError("no hay misiones activas o recientes de Agent Mode")
        return missions[-1].mission_id

    def bridge(self) -> DesktopRuntimeBridge:
        return self._bridge

    def voice_state(self):
        return self._voice.status(lightweight=True)

    def set_voice_enabled(self, enabled: bool):
        self._invalidate_voice_cache()
        return self._voice.set_enabled(enabled)

    def set_voice_muted(self, muted: bool):
        self._invalidate_voice_cache()
        return self._voice.set_muted(muted)

    def set_voice_input_enabled(self, enabled: bool):
        self._invalidate_voice_cache()
        return self._voice.set_input_enabled(enabled)

    def set_voice_input_muted(self, muted: bool):
        self._invalidate_voice_cache()
        return self._voice.set_input_muted(muted)

    def start_voice_listening(self):
        self._invalidate_voice_cache()
        return self._voice.start_listening()

    def cancel_voice_listening(self):
        self._invalidate_voice_cache()
        return self._voice.cancel_listening()

    def stop_voice(self):
        self._invalidate_voice_cache()
        return self._voice.stop()

    def test_voice(self):
        self._invalidate_voice_cache()
        self._voice.test_voice()
        return self._voice.status(lightweight=False)

    def _loading_shell_state(self) -> DesktopShellState:
        with self._state_lock:
            return DesktopShellState(
                app_name=self._jarvis.settings.app_name,
                environment=self._jarvis.settings.environment,
                busy=True,
                activity_label=self._activity_label or "STARTING",
                performance=dict(self._performance),
                quick_actions=list(self._quick_actions),
                panel_snapshot=DesktopPanelSnapshot(
                    health_summary={
                        "aggregate_status": "starting",
                        "active_operations": 0,
                    },
                ),
                conversation=list(self._conversation),
                voice=self._voice_status(force=False),
                llm_mode=getattr(self, "_env_status", None).recommended_mode
                if getattr(self, "_env_status", None)
                else "disabled",
                llm_provider=getattr(
                    self, "_env_status", None
                ).recommended_local_provider
                or "none"
                if getattr(self, "_env_status", None)
                else "none",
                dev_runtime=self._dev_runtime_state(),
            )

    def _conversation_snapshot(self) -> list[DesktopChatMessage]:
        with self._state_lock:
            return list(self._conversation)

    def _panel_snapshot(self, *, force: bool) -> DesktopPanelSnapshot:
        now = perf_counter()
        with self._state_lock:
            if (
                not force
                and self._panel_snapshot_cache is not None
                and (now - self._panel_snapshot_cached_at)
                < self._PANEL_CACHE_TTL_SECONDS
            ):
                return self._panel_snapshot_cache
            if self._active_stream_id is not None or (
                self._busy and self._activity_label == "STREAMING"
            ):
                if self._panel_snapshot_cache is not None:
                    return self._panel_snapshot_cache
                return DesktopPanelSnapshot(
                    health_summary={
                        "aggregate_status": "streaming",
                        "active_operations": 1,
                    }
                )
        snapshot = self._panels.compose()
        cached_at = perf_counter()
        with self._state_lock:
            self._panel_snapshot_cache = snapshot
            self._panel_snapshot_cached_at = cached_at
        return snapshot

    def _voice_status(self, *, force: bool):
        now = perf_counter()
        with self._state_lock:
            if (
                not force
                and self._voice_cache is not None
                and (now - self._voice_cached_at) < self._VOICE_CACHE_TTL_SECONDS
            ):
                return self._voice_cache
            if self._active_stream_id is not None or (
                self._busy and self._activity_label == "STREAMING"
            ):
                if self._voice_cache is not None:
                    return self._voice_cache
                return DesktopVoiceState(enabled=True, speaking=False)
        status = self._voice.status(lightweight=True)
        cached_at = perf_counter()
        with self._state_lock:
            self._voice_cache = status
            self._voice_cached_at = cached_at
        return status

    def _invalidate_caches(self) -> None:
        with self._state_lock:
            self._panel_snapshot_cache = None
            self._panel_snapshot_cached_at = 0.0
            self._dev_runtime_cache = None
            self._dev_runtime_cached_at = 0.0
        self._invalidate_voice_cache()

    def _invalidate_voice_cache(self) -> None:
        with self._state_lock:
            self._voice_cache = None
            self._voice_cached_at = 0.0

    def _code_agent_service(self) -> CodeAgentRuntimeService:
        with self._state_lock:
            if self._code_agent is None:
                self._code_agent = CodeAgentRuntimeService(
                    self._jarvis.settings.resolved_workspace_root
                )
            return self._code_agent

    def _dev_runtime_state(self) -> dict[str, Any]:
        now = perf_counter()
        hybrid_status = self.llm_status()
        env_status = getattr(self, "_env_status", None)
        active_provider = hybrid_status.get("active_provider") or "pending"
        active_model = (
            hybrid_status.get("active_model") or self._jarvis.settings.local_llm_model
        )
        with self._state_lock:
            if (
                self._dev_runtime_cache is not None
                and (now - self._dev_runtime_cached_at)
                < self._DEV_RUNTIME_CACHE_TTL_SECONDS
            ):
                return dict(self._dev_runtime_cache)
            if self._active_stream_id is not None or (
                self._busy and self._activity_label == "STREAMING"
            ):
                return {
                    "llm_mode": hybrid_status["selected_mode"],
                    "llm_provider": active_provider,
                    "llm_model": active_model,
                    "local_endpoint_available": bool(
                        env_status.local_endpoint.available
                    )
                    if env_status
                    else False,
                    "status": "streaming",
                    "last_result": dict(self._last_dev_result),
                }
        web_status = build_web_search_provider(self._jarvis.settings).status()
        try:
            image_status = self._jarvis.runtime_service.image_status()
        except Exception as exc:  # noqa: BLE001
            image_status = {
                "enabled": False,
                "status": "unavailable",
                "message": str(exc),
            }
        state = {
            "llm_mode": hybrid_status["selected_mode"],
            "llm_provider": active_provider,
            "llm_model": active_model,
            "local_endpoint_available": bool(env_status.local_endpoint.available)
            if env_status
            else False,
            "web_search": web_status.model_dump(mode="json"),
            "image_runtime": image_status,
            "policy": {
                "identity": "Jarvis",
                "openai": "blocked",
                "gemini": "blocked",
                "nvidia_nim": "enabled_by_selected_mode",
                "local_llm": local_model_policy_description(),
                "online_search": "Brave + configured model",
            },
            "last_result": dict(self._last_dev_result),
        }
        sanitized = self._sanitize_dev_value(state)
        with self._state_lock:
            self._dev_runtime_cache = dict(sanitized)
            self._dev_runtime_cached_at = perf_counter()
        return sanitized

    def _run_dev_action(
        self, action_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        code = self._code_agent_service()
        task = str(payload.get("task") or "").strip()
        patch_id = str(payload.get("patch_id") or "").strip()
        llm_mode = str(payload.get("llm_mode") or "auto").strip() or "auto"
        allow_online = llm_mode in {"auto", "nvidia_api"}
        if action_id == "plan":
            if not task:
                return {
                    "status": "blocked",
                    "message": "Escribe una tarea para planear.",
                }
            return code.agent_plan(task)
        if action_id == "generate_patch":
            if not task:
                return {
                    "status": "blocked",
                    "message": "Escribe una tarea para generar patch.",
                }
            result = code.change_propose(
                task, llm_assisted=True, llm_mode=llm_mode, allow_online=allow_online
            )
            if isinstance(result, dict):
                result["applied"] = False
                result["notice"] = (
                    "Patch generado solo para revision. No se aplico automaticamente."
                )
            return result
        if action_id == "patch_list":
            return code.patch_list(limit=25)
        if action_id == "patch_show":
            if not patch_id:
                return {
                    "status": "blocked",
                    "message": "Selecciona o escribe un patch_id.",
                }
            return code.patch_show(patch_id)
        if action_id == "patch_apply":
            if not patch_id:
                return {
                    "status": "blocked",
                    "message": "Selecciona o escribe un patch_id.",
                }
            confirm = bool(payload.get("confirm", False))
            pin = str(payload.get("pin") or "") or None
            result = code.patch_apply(patch_id, confirm=confirm, pin=pin)
            if isinstance(result, dict):
                result["patches"] = code.patch_list(limit=25).get("patches", [])
            return result
        if action_id == "patch_reject":
            if not patch_id:
                return {
                    "status": "blocked",
                    "message": "Selecciona o escribe un patch_id.",
                }
            result = code.patch_reject(patch_id)
            if isinstance(result, dict):
                result["patches"] = code.patch_list(limit=25).get("patches", [])
            return result
        if action_id == "git_status":
            receipt = code.git_summary()
            return (
                receipt.model_dump(mode="json")
                if hasattr(receipt, "model_dump")
                else dict(receipt)
            )
        if action_id == "memory":
            return {"status": "ok", "summary": code.memory_summary(max_chars=2500)}
        if action_id == "doctor":
            env = detect_environment(
                local_base_url=self._jarvis.settings.local_llm_base_url,
                local_model=self._jarvis.settings.local_llm_model,
                local_model_aliases=self._jarvis.settings.local_llm_model_aliases,
                nvidia_configured=self._jarvis.nvidia_secret_manager.status().configured,
            )
            code_llm = code.llm_status()
            hybrid_llm = self.llm_status()
            web_status = build_web_search_provider(self._jarvis.settings).status()
            return {
                "status": "ok",
                "internet_available": env.internet_available,
                "local_endpoint_available": env.local_endpoint.available,
                "local_endpoint_models": env.local_endpoint.models[:8],
                "recommended_mode": env.recommended_mode,
                "recommended_local_provider": env.recommended_local_provider,
                "recommended_local_model": env.recommended_local_model,
                "llm": code_llm,
                "hybrid_llm": hybrid_llm,
                "web_search": web_status.model_dump(mode="json"),
                "policy": {
                    "openai": "blocked",
                    "gemini": "blocked",
                    "nvidia_nim": "enabled_by_selected_mode",
                    "local_llm": local_model_policy_description(),
                "online_search": "Brave + configured model",
                },
                "warnings": env.warnings,
            }
        return {"status": "blocked", "message": f"Accion dev desconocida: {action_id}"}

    def _sanitize_dev_value(self, value: Any, *, max_text: int = 6000) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).casefold()
                if any(
                    term in normalized
                    for term in (
                        "pin",
                        "password",
                        "token",
                        "api_key",
                        "apikey",
                        "secret",
                        "credential",
                        "private_key",
                    )
                ):
                    safe[key] = "[redacted]"
                else:
                    safe[key] = self._sanitize_dev_value(item, max_text=max_text)
            return safe
        if isinstance(value, list):
            return [
                self._sanitize_dev_value(item, max_text=max_text) for item in value[:50]
            ]
        if isinstance(value, str):
            lowered = value.casefold()
            if any(
                term in lowered
                for term in (
                    ".env",
                    "api_key",
                    "apikey",
                    "password",
                    "token",
                    "secret",
                    "credential",
                    "private key",
                    "-----begin",
                )
            ):
                return "[redacted]"
            if len(value) > max_text:
                return value[:max_text] + "\n...[truncated]"
            return value
        return value

    def _summarize_dev_action(self, action_id: str, result: dict[str, Any]) -> str:
        status = str(result.get("status") or result.get("message") or "ok")
        if action_id == "generate_patch":
            patch = result.get("patch") if isinstance(result.get("patch"), dict) else {}
            patch_id = (
                result.get("patch_id") or patch.get("patch_id") or patch.get("id")
            )
            if patch_id:
                return f"Patch generado para revision: {patch_id}. No se aplico automaticamente."
            return f"Generacion de patch: {status}."
        if action_id == "patch_apply":
            return f"Apply patch: {status}. {result.get('message', '')}".strip()
        if action_id == "git_status":
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            git = data.get("git") if isinstance(data.get("git"), dict) else data
            branch = git.get("branch") if isinstance(git, dict) else None
            changed = git.get("changed_files") if isinstance(git, dict) else None
            return f"Git status listo. Branch: {branch or 'n/a'}. Cambios: {len(changed) if isinstance(changed, list) else 'n/a'}."
        if action_id == "memory":
            return "Resumen de memoria listo."
        if action_id == "doctor":
            return f"Doctor listo. Modo recomendado: {result.get('recommended_mode', 'n/a')}."
        return str(result.get("message") or f"Accion {action_id} completada.")

    def _set_busy_state(self, busy: bool, activity_label: str) -> None:
        with self._state_lock:
            self._busy = busy
            self._activity_label = activity_label

    def note_ui_refresh(self, *, applied: bool) -> None:
        with self._state_lock:
            key = "ui_refresh_applied" if applied else "ui_refresh_skipped"
            self._performance[key] = int(self._performance.get(key) or 0) + 1

    def _handle_voice_command(
        self, text: str, correlation_id: str | None = None, metadata: dict | None = None
    ) -> None:
        enriched_metadata = {
            "surface": "desktop_voice",
            "pipeline": "mouth_to_hands_eyes",
        }
        if metadata:
            enriched_metadata.update(metadata)
        self.send_chat(
            text,
            source="voice",
            correlation_id=correlation_id,
            metadata=enriched_metadata,
        )
        self._invalidate_voice_cache()

    @staticmethod
    def _summarize_quick_action(action_id: str, result: dict) -> str:
        if action_id == "research.run":
            return summarize_research_task(result)
        if action_id == "writing.continue":
            return summarize_writing_receipt(result)
        if action_id == "system.status":
            return result.get("message") or "Estado del system runtime actualizado."
        if action_id == "autonomy.control":
            return summarize_autonomy_view(result)
        if action_id in {
            "ops.diagnostics",
            "ops.retention",
            "ops.recover",
            "ops.reset_breaker",
        }:
            return result.get("message") or f"Operacion {action_id} completada."
        if action_id == "unity.bridge":
            return result.get("message") or "Estado de Unity Bridge actualizado."
        if action_id == "indexing.run":
            return result.get("message") or "Indexing ejecutado."
        if "resolved_target" in result or "status" in result:
            return summarize_system_operation(result)
        return result.get("message") or f"Operacion {action_id} completada."

    @staticmethod
    def _extract_model_latency_ms(result: dict) -> float | None:
        latency = result.get("latency_ms")
        if latency is not None:
            try:
                return round(float(latency), 2)
            except (TypeError, ValueError):
                return None
        metadata = result.get("metadata") or {}
        if isinstance(metadata, dict):
            candidate = metadata.get("latency_ms")
            if candidate is not None:
                try:
                    return round(float(candidate), 2)
                except (TypeError, ValueError):
                    return None
        return None
