from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock, RLock
from time import perf_counter
from uuid import uuid4

from .actions import DesktopQuickActionExecutor, build_quick_actions
from .base import DesktopChatMessage, DesktopPanelSnapshot, DesktopShellState
from .bridge import DesktopRuntimeBridge
from .chat import DesktopChatEngine
from .panels import DesktopPanelComposer
from .voice import DesktopVoiceController
from jarvis.services import (
    summarize_autonomy_view,
    summarize_research_task,
    summarize_system_operation,
    summarize_writing_receipt,
)


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
            "last_first_feedback_ms": 0.0,
            "last_shell_state_ms": None,
            "last_quick_action_ms": None,
            "requests_completed": 0,
        }
        self._panel_snapshot_cache: DesktopPanelSnapshot | None = None
        self._panel_snapshot_cached_at = 0.0
        self._voice_cache = None
        self._voice_cached_at = 0.0

    def is_ready(self) -> bool:
        return bool(self._jarvis.started)

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
            self._performance["requests_completed"] = int(self._performance["requests_completed"] or 0) + 1
            self._panel_snapshot_cache = response.panel_snapshot or None
            self._panel_snapshot_cached_at = perf_counter() if self._panel_snapshot_cache is not None else 0.0
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
            existing = self._request_futures.get(request_id)
            if existing is not None and not existing.done():
                return existing
        future = self._executor.submit(self.send_chat, text, source=source, correlation_id=request_id, metadata=metadata)
        with self._state_lock:
            self._request_futures[request_id] = future

        def _cleanup(_future: Future, *, key: str = request_id) -> None:
            with self._state_lock:
                current = self._request_futures.get(key)
                if current is _future:
                    self._request_futures.pop(key, None)

        future.add_done_callback(_cleanup)
        return future

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

    def bridge(self) -> DesktopRuntimeBridge:
        return self._bridge

    def voice_state(self):
        return self._voice.status()

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
        status = self._voice.status()
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

    def _set_busy_state(self, busy: bool, activity_label: str) -> None:
        with self._state_lock:
            self._busy = busy
            self._activity_label = activity_label

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
