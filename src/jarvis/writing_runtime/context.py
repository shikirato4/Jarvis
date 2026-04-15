from __future__ import annotations

from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.ui_automation.base import FocusWindowRequest
from jarvis.integrations import WordCOMBackend, WordCOMError, WordCOMUnavailableError, WordNoActiveDocumentError

from .models import WritingContext


class WritingContextResolver:
    def __init__(self, semantic_memory, vision_runtime, ui_automation, word_backend: WordCOMBackend | None = None, logger=None) -> None:
        self._semantic_memory = semantic_memory
        self._vision_runtime = vision_runtime
        self._ui_automation = ui_automation
        self._word_backend = word_backend
        self._logger = logger

    def detect(self, *, query: str, collection_name: str | None, correlation_id: str, target_window: str | None = None) -> WritingContext:
        if self._should_use_word_com(target_window=target_window):
            context = self._detect_word_context(query=query, collection_name=collection_name, correlation_id=correlation_id, target_window=target_window)
            if context is not None:
                return context
        active = self._ui_automation.active_window(correlation_id=correlation_id).active_window
        if target_window and active and target_window.casefold() not in active.title.casefold():
            active = self._ui_automation.focus_window(FocusWindowRequest(target=target_window), correlation_id=correlation_id).active_window
        selected_text = self._extract_selected_text()
        recent_text = self._extract_recent_text()
        visible_text = selected_text or self._extract_visible_text(correlation_id=correlation_id)
        semantic_context = self._retrieve_semantic_context(query=query, collection_name=collection_name, correlation_id=correlation_id)
        combined = "\n".join(part for part in (selected_text, recent_text, visible_text, semantic_context) if part).strip()
        return WritingContext(
            window_title=active.title if active else None,
            application_name=self._application_name(active.title if active else None),
            document_title=active.title if active else None,
            recent_text=selected_text or recent_text,
            visible_text=visible_text,
            semantic_context=semantic_context,
            combined_context=combined,
            source_confidence=0.85 if combined else 0.0,
            metadata={"active_window": active.model_dump(mode="json") if active else None, "selection_text": selected_text},
        )

    def _detect_word_context(
        self,
        *,
        query: str,
        collection_name: str | None,
        correlation_id: str,
        target_window: str | None,
    ) -> WritingContext | None:
        if self._word_backend is None:
            return None
        try:
            document_text = self._word_backend.read_active_document_text()
            metadata = self._word_backend.get_document_metadata()
        except WordNoActiveDocumentError:
            raise
        except WordCOMUnavailableError:
            self._log_word_fallback("La integración COM de Word no estuvo disponible.", target_window=target_window)
            return None
        except WordCOMError as exc:
            self._log_word_fallback(str(exc), target_window=target_window)
            return None

        semantic_context = self._retrieve_semantic_context(query=query, collection_name=collection_name, correlation_id=correlation_id)
        visible_text = self._truncate_word_context(document_text, limit=12000)
        recent_text = self._truncate_word_context(document_text, limit=4000)
        combined = "\n".join(part for part in (visible_text, semantic_context) if part).strip()
        return WritingContext(
            window_title=metadata.get("window_title") or "Word",
            application_name="word",
            document_title=metadata.get("name") or metadata.get("full_name"),
            recent_text=recent_text,
            visible_text=visible_text,
            semantic_context=semantic_context,
            combined_context=combined,
            source_confidence=0.95 if combined else 0.0,
            metadata={"integration": "word_com", "word_document": metadata},
        )

    def _extract_recent_text(self) -> str:
        backend = getattr(self._ui_automation, "_backend", None)
        typed = getattr(backend, "typed_text", "") if backend is not None else ""
        return typed[-2000:]

    def _extract_selected_text(self) -> str:
        backend = getattr(self._ui_automation, "_backend", None)
        if backend is None:
            return ""
        copier = getattr(backend, "copy_selection_text", None)
        if callable(copier):
            try:
                return str(copier() or "")[-4000:]
            except Exception:  # noqa: BLE001
                pass
        selected = getattr(backend, "typed_text", "")
        return str(selected or "")[-4000:]

    def _extract_visible_text(self, *, correlation_id: str) -> str:
        try:
            receipt = self._vision_runtime.extract_text({"capture": {"target_type": "active_window"}, "correlation_id": correlation_id})
            return receipt.ocr_result.text if receipt.ocr_result else ""
        except Exception:
            return ""

    def _retrieve_semantic_context(self, *, query: str, collection_name: str | None, correlation_id: str) -> str:
        settings = getattr(self._semantic_memory, "_settings", None)
        if settings is not None and (not getattr(settings, "ollama_enabled", True) or not getattr(settings, "embeddings_enabled", True)):
            return ""
        try:
            context = self._semantic_memory.retrieve_context(
                SemanticSearchQuery(query=query, collection_name=collection_name, top_k=4, correlation_id=correlation_id)
            )
            return "\n".join(chunk.content for chunk in context.chunks[:4])
        except Exception:
            return ""

    @staticmethod
    def _application_name(title: str | None) -> str | None:
        lowered = (title or "").casefold()
        if "word" in lowered:
            return "word"
        if "docs" in lowered or "browser" in lowered:
            return "browser"
        if "editor" in lowered:
            return "editor"
        return None

    def _should_use_word_com(self, *, target_window: str | None) -> bool:
        if self._word_backend is None:
            return False
        if self._word_backend.detect_if_word_target(target_hint=target_window):
            return True
        try:
            active = self._ui_automation.active_window(correlation_id="writing-word-detect").active_window
        except Exception:  # noqa: BLE001
            return False
        if active is None:
            return False
        return self._word_backend.detect_if_word_target(window_title=active.title, target_hint=target_window)

    def _log_word_fallback(self, reason: str, *, target_window: str | None) -> None:
        if self._logger is None:
            return
        self._logger.warning("word_com_fallback_to_ui", extra={"reason": reason, "target_window": target_window})

    @staticmethod
    def _truncate_word_context(text: str, *, limit: int) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[-limit:]
