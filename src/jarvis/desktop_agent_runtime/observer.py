from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from .memory import DesktopAgentMemoryManager
from .models import DesktopAgentObservation, DesktopAgentPhase, DesktopAgentTarget, DesktopWorldState


class DesktopAgentObserver:
    def __init__(self, *, runtime, ui_backend, memory: DesktopAgentMemoryManager) -> None:
        self._runtime = runtime
        self._ui_backend = ui_backend
        self._memory = memory

    def observe(self, world: DesktopWorldState, *, phase: DesktopAgentPhase) -> DesktopWorldState:
        active_receipt = self._runtime.ui_active_window()
        awareness = self._runtime.vision_describe_active_window()
        active_window = active_receipt.active_window
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
                "awareness_summary": awareness.data.get("summary") or (awareness.awareness_result.summary if awareness.awareness_result else ""),
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
        world.updated_at = datetime.now(timezone.utc)
        return self._memory.append_observation(world, observation)

    @staticmethod
    def _derive_context_signals(
        *,
        active_window,
        known_windows,
        visible_text: str,
        selection_text: str,
        detected_targets: Iterable[DesktopAgentTarget],
        awareness_summary: str,
    ) -> list[str]:
        signals: list[str] = []
        lowered_text = visible_text.casefold()
        lowered_summary = awareness_summary.casefold()
        if active_window is not None:
            signals.append(f"window_title:{active_window.title.casefold()}")
            if active_window.process_name:
                signals.append(f"process:{active_window.process_name.casefold()}")
            if "word" in active_window.title.casefold():
                signals.append("word_document_active")
            if any(browser in active_window.title.casefold() for browser in ("chrome", "opera", "edge")):
                signals.append("browser_active")
        for token in ("youtube", "buscar", "search", "resultados", "results", "documento", "enviar", "send"):
            if token in lowered_text or token in lowered_summary:
                signals.append(f"content:{token}")
        if selection_text:
            signals.append("selection_available")
        for window in known_windows:
            signals.append(f"known_window:{window.title.casefold()}")
        for target in detected_targets:
            label = (target.label or "").casefold().strip()
            kind = (target.kind or "").casefold().strip()
            if label:
                signals.append(f"target:{label}")
            if kind:
                signals.append(f"target_kind:{kind}")
            if kind == "input":
                signals.append("editable_input_visible")
            if label and ("buscar" in label or "search" in label):
                signals.append("search_field_visible")
            if label and ("enviar" in label or "send" in label):
                signals.append("send_button_visible")
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
