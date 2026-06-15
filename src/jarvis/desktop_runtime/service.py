from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Lock, RLock, Timer
from time import perf_counter
from typing import Any
from uuid import uuid4

from jarvis.code_agent_runtime import CodeAgentRuntimeService
from jarvis.environment import detect_environment
from .actions import DesktopQuickActionExecutor, build_quick_actions
from .base import DesktopChatMessage, DesktopPanelSnapshot, DesktopShellState
from .bridge import DesktopRuntimeBridge
from .chat import DesktopChatEngine
from .panels import DesktopPanelComposer
from .voice import DesktopVoiceController
from jarvis.identity import sanitize_assistant_identity
from jarvis.ollama_diagnostics import classify_local_model_error, local_model_failure_message
from jarvis.services import (
    summarize_autonomy_view,
    summarize_research_task,
    summarize_system_operation,
    summarize_writing_receipt,
)
from jarvis.web_search import build_web_search_provider, should_use_web_search


class DesktopRuntimeService:
    _PANEL_CACHE_TTL_SECONDS = 1.0
    _VOICE_CACHE_TTL_SECONDS = 0.25

    def __init__(self, jarvis_app) -> None:
        self._jarvis = jarvis_app
        self._bridge = DesktopRuntimeBridge(jarvis_app)
        self._panels = DesktopPanelComposer(self._bridge)
        self._actions = DesktopQuickActionExecutor(self._bridge)
        self._chat = DesktopChatEngine(self._bridge, self._panels, self._actions, jarvis_app.settings)
        self._voice = DesktopVoiceController(
            jarvis_app.voice_runtime_service,
            jarvis_app.settings,
            self._handle_voice_command,
        )
        self._logger = logging.getLogger("jarvis.desktop.service")
        self._state_lock = RLock()
        self._request_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="jarvis-desktop")
        self._quick_actions = build_quick_actions()
        self._request_futures: dict[str, Future] = {}
        self._completed_chat_requests: dict[str, object] = {}
        self._active_stream_id: str | None = None
        self._active_stream_cancel: Event | None = None
        self._conversation: list[DesktopChatMessage] = [
            DesktopChatMessage(
                message_id=str(uuid4()),
                role="system",
                content="JARVIS desktop interface online. All runtimes remain delegated to the existing backend.",
            )
        ]
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
        self._voice_cache = None
        self._voice_cached_at = 0.0
        self._code_agent: CodeAgentRuntimeService | None = None
        self._last_dev_result: dict[str, Any] = {
            "status": "idle",
            "action": "none",
            "summary": "Code Agent listo. Genera patches revisables; nada se aplica automaticamente.",
        }

    def is_ready(self) -> bool:
        return bool(self._jarvis.started)

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
                self._logger.info("desktop_startup_completed", extra={"startup_ms": startup_ms})
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
            state = DesktopShellState(
                app_name=snapshot.app_name,
                environment=snapshot.environment,
                busy=self._busy,
                activity_label=self._activity_label if self._busy else "IDLE",
                performance=dict(self._performance),
                quick_actions=list(self._quick_actions),
                panel_snapshot=self._panel_snapshot(force=force),
                conversation=self._conversation_snapshot(),
                voice=self._voice_status(force=force),
                llm_mode=getattr(self, "_env_status", None).recommended_mode if getattr(self, "_env_status", None) else "disabled",
                llm_provider=getattr(self, "_env_status", None).recommended_local_provider or "none" if getattr(self, "_env_status", None) else "none",
                dev_runtime=self._dev_runtime_state(),
            )
        elapsed_ms = (perf_counter() - started_at) * 1000
        with self._state_lock:
            self._performance["last_shell_state_ms"] = round(elapsed_ms, 2)
            state.performance = dict(self._performance)
        return state

    def refresh(self) -> DesktopShellState:
        return self.shell_state()

    def send_chat(self, text: str, *, source: str = "text", correlation_id: str | None = None, metadata: dict | None = None):
        self.start_backend()
        if correlation_id is not None:
            with self._state_lock:
                cached = self._completed_chat_requests.get(correlation_id)
            if cached is not None:
                self._logger.info("desktop_chat_deduplicated_completed", extra={"correlation_id": correlation_id, "source": source})
                return cached
        queued_at = perf_counter()
        user_message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="user",
            content=text,
            metadata={"source": source, "correlation_id": correlation_id, **(metadata or {})},
        )
        with self._state_lock:
            self._conversation.append(user_message)
            self._performance["last_first_feedback_ms"] = round((perf_counter() - queued_at) * 1000, 2)
        self._set_busy_state(True, "PROCESSING")
        self._invalidate_caches()
        started_at = perf_counter()
        with self._request_lock:
            response = self._chat.handle(text)
        elapsed_ms = (perf_counter() - started_at) * 1000
        response.message.metadata.setdefault("source", source)
        if correlation_id is not None:
            response.message.metadata.setdefault("correlation_id", correlation_id)
        with self._state_lock:
            self._conversation.append(response.message)
            self._performance["last_request_ms"] = round(elapsed_ms, 2)
            self._performance["model_latency_ms"] = self._extract_model_latency_ms(response.raw_result)
            self._performance["requests_completed"] = int(self._performance["requests_completed"] or 0) + 1
            self._panel_snapshot_cache = response.panel_snapshot or None
            self._panel_snapshot_cached_at = perf_counter() if self._panel_snapshot_cache is not None else 0.0
            if correlation_id is not None:
                self._completed_chat_requests[correlation_id] = response
                while len(self._completed_chat_requests) > 32:
                    oldest = next(iter(self._completed_chat_requests))
                    self._completed_chat_requests.pop(oldest, None)
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
                self._voice.speak_literal(response.spoken_content or response.message.content)
            else:
                self._voice.speak_response(response.spoken_content or response.message.content)
        return response

    def send_chat_async(self, text: str, *, source: str = "text", correlation_id: str | None = None, metadata: dict | None = None) -> Future:
        request_id = correlation_id or f"desktop-chat-{uuid4().hex[:12]}"
        with self._state_lock:
            completed = self._completed_chat_requests.get(request_id)
            if completed is not None:
                future = Future()
                future.set_result(completed)
                return future
            existing = self._request_futures.get(request_id)
            if existing is not None and not existing.done():
                self._logger.info("desktop_chat_deduplicated_inflight", extra={"correlation_id": request_id, "source": source})
                return existing
        use_stream = bool((metadata or {}).get("stream"))
        future: Future = Future()
        with self._state_lock:
            self._request_futures[request_id] = future
        self._set_busy_state(True, "STREAMING" if use_stream else "PROCESSING")

        def _cleanup(*, key: str = request_id) -> None:
            with self._state_lock:
                current = self._request_futures.get(key)
                if current is future:
                    self._request_futures.pop(key, None)

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
                worker_future = self._executor.submit(self.send_chat_streaming, text, source=source, correlation_id=request_id, metadata=metadata)
            else:
                worker_future = self._executor.submit(self.send_chat, text, source=source, correlation_id=request_id, metadata=metadata)
            worker_future.add_done_callback(_copy_result)

        Timer(0.01, _start_worker).start()
        return future

    def send_chat_streaming(self, text: str, *, source: str = "text", correlation_id: str | None = None, metadata: dict | None = None):
        self.start_backend()
        correlation_id = correlation_id or f"desktop-stream-{uuid4().hex[:12]}"
        cancel_event = Event()
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
            metadata={"source": source, "correlation_id": correlation_id, **(metadata or {})},
        )
        assistant_message = DesktopChatMessage(
            message_id=str(uuid4()),
            role="assistant",
            content="Pensando localmente con gpt-oss...",
            metadata={"source": source, "correlation_id": correlation_id, "streaming": True, "status": "thinking"},
        )
        with self._state_lock:
            self._conversation.append(user_message)
            self._conversation.append(assistant_message)
            self._performance["last_first_feedback_ms"] = round((perf_counter() - queued_at) * 1000, 2)
        self._set_busy_state(True, "STREAMING")
        self._invalidate_caches()
        started_at = perf_counter()
        full_text = ""
        raw_result: dict[str, Any] = {"status": "streaming", "streaming": True}
        spoken_content: str | None = None
        try:
            with self._request_lock:
                if should_use_web_search(text, mode="auto"):
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        "Buscando fuentes con Brave...",
                        metadata={"streaming": True, "status": "searching_web"},
                    )
                prepared = self._chat.prepare_streaming_request(text.strip())
                if prepared is None:
                    response = self._chat.handle(text)
                    raw_result = response.raw_result
                    spoken_content = response.spoken_content or response.message.content
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        response.message.content,
                        metadata={"streaming": False, "status": "completed", "result": raw_result},
                    )
                    return response
                if hasattr(prepared, "message"):
                    response = prepared
                    self._replace_conversation_message(assistant_message.message_id, response.message.content, metadata={"streaming": False, "status": "completed", "result": response.raw_result})
                    raw_result = response.raw_result
                    spoken_content = response.spoken_content or response.message.content
                    return response
                request, web_result = prepared
                if web_result is not None:
                    self._replace_conversation_message(
                        assistant_message.message_id,
                        f"Encontre {len(web_result.hits)} fuentes. Redactando localmente con gpt-oss...",
                        metadata={"streaming": True, "status": "synthesizing", "web_search": web_result.model_dump(mode="json")},
                    )
                    raw_result["web_search"] = web_result.model_dump(mode="json")
                chunks = 0
                first_token_ms: float | None = None
                for chunk in self._jarvis.runtime_service.stream_model(request, cancel_check=cancel_event.is_set):
                    if not self._is_active_stream(correlation_id) or cancel_event.is_set():
                        raw_result["status"] = "cancelled"
                        return self._stream_response(assistant_message, "", raw_result, cancelled=True)
                    if chunk.error:
                        reason = classify_local_model_error(Exception(chunk.error))
                        message = local_model_failure_message(reason, web_sources=len(getattr(web_result, "hits", []) or []))
                        raw_result.update({"status": "error", "error": reason})
                        self._replace_conversation_message(assistant_message.message_id, message, metadata={"streaming": False, "status": "error", "result": raw_result})
                        return self._stream_response(assistant_message, message, raw_result)
                    if chunk.text:
                        if first_token_ms is None:
                            first_token_ms = round((perf_counter() - started_at) * 1000, 2)
                        chunks += 1
                        full_text += chunk.text
                        self._replace_conversation_message(
                            assistant_message.message_id,
                            sanitize_assistant_identity(full_text),
                            metadata={"streaming": True, "status": "streaming", "chunks": chunks},
                        )
                    if chunk.done:
                        raw_result.update(
                            {
                                "status": "completed",
                                "provider_name": chunk.metadata.get("provider"),
                                "model_name": chunk.metadata.get("model"),
                                "content": sanitize_assistant_identity(full_text),
                                "chunks": chunks,
                                "first_token_ms": first_token_ms,
                                "latency_ms": round((perf_counter() - started_at) * 1000, 2),
                            }
                        )
                        break
            final_text = sanitize_assistant_identity(full_text).strip() or "No obtuve una salida valida del sistema, pero sigo listo para responder."
            spoken_content = final_text
            self._replace_conversation_message(assistant_message.message_id, final_text, metadata={"streaming": False, "status": "completed", "result": raw_result})
            return self._stream_response(assistant_message, final_text, raw_result)
        except Exception as exc:  # noqa: BLE001
            reason = classify_local_model_error(exc)
            message = local_model_failure_message(reason, web_sources=0)
            raw_result.update({"status": "error", "error": reason})
            self._replace_conversation_message(assistant_message.message_id, message, metadata={"streaming": False, "status": "error", "result": raw_result})
            return self._stream_response(assistant_message, message, raw_result)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            with self._state_lock:
                self._performance["last_request_ms"] = round(elapsed_ms, 2)
                self._performance["model_latency_ms"] = raw_result.get("latency_ms")
                self._performance["requests_completed"] = int(self._performance["requests_completed"] or 0) + 1
                if self._active_stream_id == correlation_id:
                    self._active_stream_id = None
                    self._active_stream_cancel = None
                self._completed_chat_requests[correlation_id] = self._stream_response(assistant_message, spoken_content or full_text, raw_result)
            self._set_busy_state(False, "IDLE")
            if raw_result.get("status") == "completed" and spoken_content:
                self._voice.speak_response(spoken_content)

    def _replace_conversation_message(self, message_id: str, content: str, *, metadata: dict[str, Any] | None = None) -> None:
        with self._state_lock:
            for index, message in enumerate(self._conversation):
                if message.message_id == message_id:
                    merged = dict(message.metadata)
                    if metadata:
                        merged.update(metadata)
                    self._conversation[index] = message.model_copy(update={"content": content, "metadata": merged})
                    break
        self._invalidate_caches()

    def _is_active_stream(self, correlation_id: str) -> bool:
        with self._state_lock:
            return self._active_stream_id == correlation_id

    @staticmethod
    def _stream_response(message: DesktopChatMessage, content: str, raw_result: dict[str, Any], *, cancelled: bool = False):
        final_message = message.model_copy(update={"content": content, "metadata": {**message.metadata, "result": raw_result}})
        from .base import DesktopChatResponse

        return DesktopChatResponse(message=final_message, spoken_content=None if cancelled else content, spoken_mode="prepared", raw_result=raw_result)

    def execute_quick_action(self, action_id: str, *, payload: dict | None = None) -> dict:
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
        self._logger.info("desktop_quick_action_completed", extra={"action_id": action_id, "request_ms": elapsed_ms})
        return result

    def execute_quick_action_async(self, action_id: str, *, payload: dict | None = None) -> Future:
        return self._executor.submit(self.execute_quick_action, action_id, payload=payload)

    def execute_dev_action(self, action_id: str, *, payload: dict | None = None) -> dict:
        self.start_backend()
        payload = payload or {}
        self._set_busy_state(True, f"CODE {action_id.upper()}")
        started_at = perf_counter()
        try:
            result = self._run_dev_action(action_id, payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("desktop_dev_action_failed", extra={"action_id": action_id, "exception_type": type(exc).__name__})
            result = {"status": "failed", "action": action_id, "message": str(exc), "error_type": type(exc).__name__}
        result = self._sanitize_dev_value(result)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        if isinstance(result, dict):
            result.setdefault("action", action_id)
            result["elapsed_ms"] = elapsed_ms
        summary = self._summarize_dev_action(action_id, result if isinstance(result, dict) else {"result": result})
        with self._state_lock:
            self._last_dev_result = result if isinstance(result, dict) else {"status": "ok", "action": action_id, "result": result}
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
        self._logger.info("desktop_dev_action_completed", extra={"action_id": action_id, "request_ms": elapsed_ms})
        return self._last_dev_result

    def execute_dev_action_async(self, action_id: str, *, payload: dict | None = None) -> Future:
        return self._executor.submit(self.execute_dev_action, action_id, payload=payload)

    def confirm_latest_agent_action(self) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "AGENT CONFIRM")
        started_at = perf_counter()
        try:
            mission_id = self._latest_desktop_agent_mission_id()
            receipt = self._jarvis.runtime_service.desktop_agent_confirm(mission_id)
            result = receipt.model_dump(mode="json")
            summary = str(result.get("summary") or "Accion confirmada y ejecutada por Agent Mode.")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_agent_confirm_failed",
                extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            result = {"status": "failed", "message": self._sanitize_dev_value(str(exc)), "error_type": type(exc).__name__}
            summary = f"No pude confirmar la accion del agente: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={"agent_action": "confirm", "result": self._sanitize_dev_value(result)},
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def confirm_latest_agent_action_async(self) -> Future:
        return self._executor.submit(self.confirm_latest_agent_action)

    def stop_latest_agent(self) -> dict[str, Any]:
        self.start_backend()
        self._set_busy_state(True, "AGENT STOP")
        started_at = perf_counter()
        try:
            mission_id = self._latest_desktop_agent_mission_id()
            receipt = self._jarvis.runtime_service.desktop_agent_abort(mission_id, reason="stop agent requested from desktop UI")
            result = receipt.model_dump(mode="json")
            summary = str(result.get("summary") or "Agent Mode detenido.")
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "desktop_agent_stop_failed",
                extra={"exception_type": type(exc).__name__, "exception_message": str(exc)},
            )
            result = {"status": "failed", "message": self._sanitize_dev_value(str(exc)), "error_type": type(exc).__name__}
            summary = f"No pude detener el agente: {self._sanitize_dev_value(str(exc))}"
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        result["elapsed_ms"] = elapsed_ms
        with self._state_lock:
            self._conversation.append(
                DesktopChatMessage(
                    message_id=str(uuid4()),
                    role="assistant",
                    content=summary,
                    metadata={"agent_action": "stop", "result": self._sanitize_dev_value(result)},
                )
            )
            self._performance["last_quick_action_ms"] = elapsed_ms
        self._invalidate_caches()
        self._set_busy_state(False, "IDLE")
        return result

    def stop_latest_agent_async(self) -> Future:
        return self._executor.submit(self.stop_latest_agent)

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
                    health_summary={"aggregate_status": "starting", "active_operations": 0},
                ),
                conversation=list(self._conversation),
                voice=self._voice_status(force=False),
                llm_mode=getattr(self, "_env_status", None).recommended_mode if getattr(self, "_env_status", None) else "disabled",
                llm_provider=getattr(self, "_env_status", None).recommended_local_provider or "none" if getattr(self, "_env_status", None) else "none",
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
                and (now - self._panel_snapshot_cached_at) < self._PANEL_CACHE_TTL_SECONDS
            ):
                return self._panel_snapshot_cache
        snapshot = self._panels.compose()
        cached_at = perf_counter()
        with self._state_lock:
            self._panel_snapshot_cache = snapshot
            self._panel_snapshot_cached_at = cached_at
        return snapshot

    def _voice_status(self, *, force: bool):
        now = perf_counter()
        with self._state_lock:
            if not force and self._voice_cache is not None and (now - self._voice_cached_at) < self._VOICE_CACHE_TTL_SECONDS:
                return self._voice_cache
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
        self._invalidate_voice_cache()

    def _invalidate_voice_cache(self) -> None:
        with self._state_lock:
            self._voice_cache = None
            self._voice_cached_at = 0.0

    def _code_agent_service(self) -> CodeAgentRuntimeService:
        with self._state_lock:
            if self._code_agent is None:
                self._code_agent = CodeAgentRuntimeService(self._jarvis.settings.workspace_root)
            return self._code_agent

    def _dev_runtime_state(self) -> dict[str, Any]:
        env_status = getattr(self, "_env_status", None)
        web_status = build_web_search_provider().status()
        state = {
            "llm_mode": env_status.recommended_mode if env_status else "disabled",
            "llm_provider": env_status.recommended_local_provider if env_status else "none",
            "llm_model": env_status.recommended_local_model if env_status else None,
            "ollama_available": bool(env_status.ollama.available) if env_status else False,
            "web_search": web_status.model_dump(mode="json"),
            "policy": {
                "identity": "Jarvis",
                "openai": "blocked",
                "gemini": "blocked",
                "online_llm": "disabled",
                "online_search": "Brave + local Ollama",
            },
            "last_result": dict(self._last_dev_result),
        }
        return self._sanitize_dev_value(state)

    def _run_dev_action(self, action_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        code = self._code_agent_service()
        task = str(payload.get("task") or "").strip()
        patch_id = str(payload.get("patch_id") or "").strip()
        llm_mode = str(payload.get("llm_mode") or "auto").strip() or "auto"
        allow_online = llm_mode == "online"
        if action_id == "plan":
            if not task:
                return {"status": "blocked", "message": "Escribe una tarea para planear."}
            return code.agent_plan(task)
        if action_id == "generate_patch":
            if not task:
                return {"status": "blocked", "message": "Escribe una tarea para generar patch."}
            result = code.change_propose(task, llm_assisted=True, llm_mode=llm_mode, allow_online=allow_online)
            if isinstance(result, dict):
                result["applied"] = False
                result["notice"] = "Patch generado solo para revision. No se aplico automaticamente."
            return result
        if action_id == "patch_list":
            return code.patch_list(limit=25)
        if action_id == "patch_show":
            if not patch_id:
                return {"status": "blocked", "message": "Selecciona o escribe un patch_id."}
            return code.patch_show(patch_id)
        if action_id == "patch_apply":
            if not patch_id:
                return {"status": "blocked", "message": "Selecciona o escribe un patch_id."}
            confirm = bool(payload.get("confirm", False))
            pin = str(payload.get("pin") or "") or None
            result = code.patch_apply(patch_id, confirm=confirm, pin=pin)
            if isinstance(result, dict):
                result["patches"] = code.patch_list(limit=25).get("patches", [])
            return result
        if action_id == "patch_reject":
            if not patch_id:
                return {"status": "blocked", "message": "Selecciona o escribe un patch_id."}
            result = code.patch_reject(patch_id)
            if isinstance(result, dict):
                result["patches"] = code.patch_list(limit=25).get("patches", [])
            return result
        if action_id == "git_status":
            receipt = code.git_summary()
            return receipt.model_dump(mode="json") if hasattr(receipt, "model_dump") else dict(receipt)
        if action_id == "memory":
            return {"status": "ok", "summary": code.memory_summary(max_chars=2500)}
        if action_id == "doctor":
            env = detect_environment()
            llm = code.llm_status()
            web_status = build_web_search_provider().status()
            return {
                "status": "ok",
                "internet_available": env.internet_available,
                "ollama_available": env.ollama.available,
                "ollama_models": env.ollama.models[:8],
                "recommended_mode": env.recommended_mode,
                "recommended_local_provider": env.recommended_local_provider,
                "recommended_local_model": env.recommended_local_model,
                "llm": llm,
                "web_search": web_status.model_dump(mode="json"),
                "policy": {
                    "openai": "blocked",
                    "gemini": "blocked",
                    "online_llm": "disabled",
                    "online_search": "Brave + local Ollama",
                },
                "warnings": env.warnings,
            }
        return {"status": "blocked", "message": f"Accion dev desconocida: {action_id}"}

    def _sanitize_dev_value(self, value: Any, *, max_text: int = 6000) -> Any:
        if isinstance(value, dict):
            safe: dict[str, Any] = {}
            for key, item in value.items():
                normalized = str(key).casefold()
                if any(term in normalized for term in ("pin", "password", "token", "api_key", "apikey", "secret", "credential", "private_key")):
                    safe[key] = "[redacted]"
                else:
                    safe[key] = self._sanitize_dev_value(item, max_text=max_text)
            return safe
        if isinstance(value, list):
            return [self._sanitize_dev_value(item, max_text=max_text) for item in value[:50]]
        if isinstance(value, str):
            lowered = value.casefold()
            if any(term in lowered for term in (".env", "api_key", "apikey", "password", "token", "secret", "credential", "private key", "-----begin")):
                return "[redacted]"
            if len(value) > max_text:
                return value[:max_text] + "\n...[truncated]"
            return value
        return value

    def _summarize_dev_action(self, action_id: str, result: dict[str, Any]) -> str:
        status = str(result.get("status") or result.get("message") or "ok")
        if action_id == "generate_patch":
            patch = result.get("patch") if isinstance(result.get("patch"), dict) else {}
            patch_id = result.get("patch_id") or patch.get("patch_id") or patch.get("id")
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

    def _handle_voice_command(self, text: str, correlation_id: str | None = None, metadata: dict | None = None) -> None:
        enriched_metadata = {"surface": "desktop_voice", "pipeline": "mouth_to_hands_eyes"}
        if metadata:
            enriched_metadata.update(metadata)
        self.send_chat(text, source="voice", correlation_id=correlation_id, metadata=enriched_metadata)
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
        if action_id in {"ops.diagnostics", "ops.retention", "ops.recover", "ops.reset_breaker"}:
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
