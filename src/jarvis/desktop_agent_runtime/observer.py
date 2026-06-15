from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from time import perf_counter
from typing import Callable, Iterable

from jarvis.core.errors import JarvisError

from .memory import DesktopAgentMemoryManager
from .models import DesktopAgentObservation, DesktopAgentPhase, DesktopAgentTarget, DesktopWorldState


class DesktopAgentObserver:
    _OBSERVATION_CACHE_TTL_SECONDS = 0.35

    def __init__(self, *, runtime, ui_backend, memory: DesktopAgentMemoryManager) -> None:
        self._runtime = runtime
        self._ui_backend = ui_backend
        self._memory = memory
        self._awareness_cache: tuple[tuple[str, str, str], object] | None = None
        self._observation_cache: tuple[tuple[str, str, str], float, DesktopAgentObservation] | None = None

    def observe(self, world: DesktopWorldState, *, phase: DesktopAgentPhase) -> DesktopWorldState:
        previous_window_title = world.active_window.title if world.active_window is not None else None
        active_receipt = self._runtime.ui_active_window()
        active_window = active_receipt.active_window
        cache_key = (
            active_window.title if active_window is not None else "",
            active_window.process_name if active_window is not None and active_window.process_name else "",
            getattr(self._ui_backend, "typed_text", "") or "",
        )
        if self._observation_cache is not None and self._observation_cache[0] == cache_key and (perf_counter() - self._observation_cache[1]) < self._OBSERVATION_CACHE_TTL_SECONDS:
            cached = self._observation_cache[2]
            world.phase = phase
            world.known_windows = self._ui_backend.list_windows() if hasattr(self._ui_backend, "list_windows") else ([active_window] if active_window else [])
            world.active_window = cached.active_window
            world.visible_text = cached.visible_text
            world.selection_text = cached.selection_text
            world.clipboard_text = cached.clipboard_text
            world.context_signals = cached.context_signals
            world.detected_targets = cached.detected_targets
            world.last_observation_summary = cached.summary
            world.observe_count += 1
            world.updated_at = datetime.now(timezone.utc)
            return self._memory.append_observation(world, cached)
        cached = self._awareness_cache if self._awareness_cache and self._awareness_cache[0] == cache_key else None
        if cached is not None:
            awareness, awareness_mode, awareness_errors = cached[1]
            awareness_mode = f"{awareness_mode}:cache"
        else:
            awareness, awareness_mode, awareness_errors = self._capture_awareness()
            self._awareness_cache = (cache_key, (awareness, awareness_mode, awareness_errors))
        known_windows = self._ui_backend.list_windows() if hasattr(self._ui_backend, "list_windows") else ([active_window] if active_window else [])
        visible_text = ""
        detected_targets: list[DesktopAgentTarget] = []
        selection_text = ""
        clipboard_text = ""
        if awareness.awareness_result is not None:
            visible_text = "\n".join(block.text for block in awareness.awareness_result.text_blocks).strip()
            detected_targets = [
                DesktopAgentTarget(
                    label=item.label or item.text or "",
                    kind=item.kind.value if hasattr(item.kind, "value") else str(item.kind),
                    confidence=item.confidence,
                    metadata=item.metadata,
                )
                for item in awareness.awareness_result.elements
            ]
        backend_text = getattr(self._ui_backend, "typed_text", "")
        if backend_text:
            visible_text = f"{visible_text}\n{backend_text}".strip()
        if hasattr(self._ui_backend, "copy_selection_text"):
            try:
                selection_text = str(self._ui_backend.copy_selection_text() or "").strip()
            except Exception:  # noqa: BLE001
                selection_text = ""
        if selection_text:
            clipboard_text = selection_text
        context_signals = self._derive_context_signals(
            active_window=active_window,
            previous_window_title=previous_window_title,
            goal=world.current_goal,
            known_windows=known_windows,
            visible_text=visible_text,
            selection_text=selection_text,
            detected_targets=detected_targets,
            awareness_summary=awareness.data.get("summary") or (awareness.awareness_result.summary if awareness.awareness_result else ""),
        )
        summary = self._summarize_observation(
            active_window=active_window,
            visible_text=visible_text,
            detected_targets=detected_targets,
            context_signals=context_signals,
        )
        observation = DesktopAgentObservation(
            phase=phase,
            active_window=active_window,
            visible_text=visible_text,
            selection_text=selection_text,
            clipboard_text=clipboard_text,
            detected_targets=detected_targets,
            context_signals=context_signals,
            summary=summary,
            metadata={
                "known_window_count": len(known_windows),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "awareness_mode": awareness_mode,
                "awareness_errors": awareness_errors,
                "awareness_summary": awareness.data.get("summary") or (awareness.awareness_result.summary if awareness.awareness_result else ""),
                "observation_target": "active_window" if "active_window" in awareness_mode else "screen",
                "observation_status": "degraded" if awareness_errors else "ok",
                "sensitive_blocked": any(_looks_sensitive_error(str(error)) for error in awareness_errors),
                "recoverable_error": " | ".join(str(error) for error in awareness_errors[:2]) if awareness_errors else None,
            },
        )
        world.phase = phase
        world.known_windows = known_windows
        world.active_window = active_window
        world.visible_text = visible_text
        world.selection_text = selection_text
        world.clipboard_text = clipboard_text
        world.context_signals = context_signals
        world.detected_targets = detected_targets
        world.last_observation_summary = summary
        world.observe_count += 1
        world.updated_at = datetime.now(timezone.utc)
        self._observation_cache = (cache_key, perf_counter(), observation)
        return self._memory.append_observation(world, observation)

    def _capture_awareness(self):
        attempts: list[tuple[str, Callable[[], object]]] = [
            ("active_window", self._runtime.vision_describe_active_window),
            (
                "screen",
                lambda: self._runtime.vision_ui_awareness(
                    {
                        "capture": {"target_type": "screen"},
                        "include_ocr": True,
                        "include_ui_tree": True,
                        "metadata": {"source": "desktop_agent_observer", "fallback": "full_screen"},
                    }
                ),
            ),
        ]
        errors: list[str] = []
        for mode, action in attempts:
            try:
                return action(), mode, errors
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, JarvisError):
                    errors.append(f"{mode}:{exc.component}:{exc.code}:{exc.message}")
                else:
                    errors.append(f"{mode}:{type(exc).__name__}:{exc}")
        joined = " | ".join(errors) if errors else "no capture backend available"
        if any(_looks_sensitive_error(error) for error in errors):
            return (
                SimpleNamespace(
                    data={"summary": "No pude capturar esa ventana porque parece sensible o protegida."},
                    awareness_result=None,
                ),
                "degraded:sensitive_window",
                errors,
            )
        raise RuntimeError(f"desktop observation failed: {joined}")

    @staticmethod
    def _derive_context_signals(
        *,
        active_window,
        previous_window_title: str | None,
        goal: str,
        known_windows,
        visible_text: str,
        selection_text: str,
        detected_targets: Iterable[DesktopAgentTarget],
        awareness_summary: str,
    ) -> list[str]:
        signals: list[str] = []
        lowered_text = visible_text.casefold()
        lowered_summary = awareness_summary.casefold()
        lowered_goal = goal.casefold()
        if active_window is not None:
            signals.append(f"window_title:{active_window.title.casefold()}")
            if active_window.process_name:
                signals.append(f"process:{active_window.process_name.casefold()}")
            if "word" in active_window.title.casefold():
                signals.append("word_document_active")
            if "explor" in active_window.title.casefold():
                signals.append("explorer_active")
            if any(browser in active_window.title.casefold() for browser in ("chrome", "opera", "edge")):
                signals.append("browser_active")
        if previous_window_title and active_window and previous_window_title.casefold() != active_window.title.casefold():
            signals.append("window_changed")
        for token in ("youtube", "buscar", "search", "resultados", "results", "documento", "enviar", "send"):
            if token in lowered_text or token in lowered_summary:
                signals.append(f"content:{token}")
        for token in ("chrome", "word", "vscode", "youtube", "enviar", "formulario"):
            if token in lowered_goal and (token in lowered_text or token in lowered_summary):
                signals.append(f"goal_evidence:{token}")
        if selection_text:
            signals.append("selection_available")
        target_count = 0
        button_count = 0
        input_count = 0
        for window in known_windows:
            signals.append(f"known_window:{window.title.casefold()}")
        for target in detected_targets:
            target_count += 1
            label = (target.label or "").casefold().strip()
            kind = (target.kind or "").casefold().strip()
            if label:
                signals.append(f"target:{label}")
            if kind:
                signals.append(f"target_kind:{kind}")
            if kind == "input":
                input_count += 1
                signals.append("editable_input_visible")
            if label and ("buscar" in label or "search" in label):
                signals.append("search_field_visible")
            if label and ("enviar" in label or "send" in label):
                signals.append("send_button_visible")
            if kind == "button":
                button_count += 1
        if target_count:
            signals.append(f"target_count:{target_count}")
        if button_count:
            signals.append(f"button_count:{button_count}")
        if input_count:
            signals.append(f"input_count:{input_count}")
        return sorted({signal for signal in signals if signal})

    @staticmethod
    def _summarize_observation(*, active_window, visible_text: str, detected_targets: list[DesktopAgentTarget], context_signals: list[str]) -> str:
        window_label = active_window.title if active_window is not None else "sin ventana activa"
        text_preview = visible_text.replace("\n", " ").strip()[:120]
        target_labels = ", ".join(target.label for target in detected_targets[:3] if target.label)
        signal_preview = ", ".join(context_signals[:4])
        parts = [f"ventana={window_label}"]
        if text_preview:
            parts.append(f"texto='{text_preview}'")
        if target_labels:
            parts.append(f"objetivos={target_labels}")
        if signal_preview:
            parts.append(f"senales={signal_preview}")
        return " | ".join(parts)


def _looks_sensitive_error(text: str) -> bool:
    folded = text.casefold()
    return any(token in folded for token in ("sensitive", "sensible", "protected", "proteg", "blocked"))
