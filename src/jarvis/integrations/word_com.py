from __future__ import annotations

import importlib
import logging
from typing import Any


class WordCOMError(RuntimeError):
    """Base error for Word COM integration."""


class WordCOMUnavailableError(WordCOMError):
    """Raised when Word COM cannot be reached."""


class WordNoActiveDocumentError(WordCOMError):
    """Raised when Word is available but has no active document."""


class WordCOMBackend:
    _WORD_HINTS = ("word", "winword", "microsoft word")

    def __init__(self, app_provider=None, *, logger: logging.Logger | None = None) -> None:
        self._app_provider = app_provider or self._default_app_provider
        self._logger = logger or logging.getLogger("jarvis.integrations.word_com")

    def is_word_available(self) -> bool:
        try:
            self._get_application()
        except WordCOMError:
            return False
        return True

    def detect_if_word_target(
        self,
        *,
        window_title: str | None = None,
        process_name: str | None = None,
        target_hint: str | None = None,
    ) -> bool:
        candidates = (
            (window_title or "").casefold(),
            (process_name or "").casefold(),
            (target_hint or "").casefold(),
        )
        return any(any(hint in candidate for hint in self._WORD_HINTS) for candidate in candidates if candidate)

    def get_active_document(self):
        application = self._get_application()
        try:
            document = application.ActiveDocument
        except Exception as exc:  # noqa: BLE001
            raise WordNoActiveDocumentError("Word no tiene un documento activo.") from exc
        if document is None:
            raise WordNoActiveDocumentError("Word no tiene un documento activo.")
        return document

    def read_active_document_text(self) -> str:
        document = self.get_active_document()
        try:
            text = document.Content.Text
        except Exception as exc:  # noqa: BLE001
            raise WordCOMError("No pude leer el documento activo de Word.") from exc
        return self._normalize_text(text)

    def insert_text_at_cursor(self, text: str) -> dict[str, Any]:
        if not text:
            return {"inserted_at": "selection", "selection_available": False, "saved": False}
        application = self._get_application()
        document = self.get_active_document()
        selection = getattr(application, "Selection", None)
        if selection is not None:
            try:
                range_end = int(selection.Range.End)
                content_end = int(document.Content.End)
                if self._ends_with_document_marker(document) and range_end >= content_end:
                    insertion_point = max(content_end - 1, 0)
                else:
                    insertion_point = max(range_end, 0)
                document.Range(insertion_point, insertion_point).Text = text
                return {"inserted_at": "selection", "selection_available": True, "saved": False}
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("word_com_insert_at_cursor_failed", extra={"error": str(exc)})
        return self.append_text(text)

    def append_text(self, text: str) -> dict[str, Any]:
        document = self.get_active_document()
        try:
            content_end = int(document.Content.End)
            insertion_point = max(content_end - 1, 0)
            document.Range(insertion_point, insertion_point).Text = text
        except Exception as exc:  # noqa: BLE001
            raise WordCOMError("No pude escribir en el documento activo de Word.") from exc
        return {"inserted_at": "document_end", "selection_available": False, "saved": False}

    def save_document(self) -> None:
        document = self.get_active_document()
        try:
            document.Save()
        except Exception as exc:  # noqa: BLE001
            raise WordCOMError("No pude guardar el documento activo de Word.") from exc

    def get_document_metadata(self) -> dict[str, Any]:
        document = self.get_active_document()
        return {
            "name": getattr(document, "Name", None),
            "full_name": getattr(document, "FullName", None),
            "saved": getattr(document, "Saved", None),
            "window_title": self._document_window_title(document),
            "character_count": len(self.read_active_document_text()),
        }

    def _get_application(self):
        try:
            application = self._app_provider()
        except WordCOMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WordCOMUnavailableError("La integración COM de Word no estuvo disponible.") from exc
        if application is None:
            raise WordCOMUnavailableError("La integración COM de Word no estuvo disponible.")
        return application

    def _default_app_provider(self):
        try:
            win32_client = importlib.import_module("win32com.client")
        except ImportError as exc:
            raise WordCOMUnavailableError("La integración COM de Word no estuvo disponible.") from exc

        get_active_object = getattr(win32_client, "GetActiveObject", None)
        if callable(get_active_object):
            try:
                return get_active_object("Word.Application")
            except Exception as exc:  # noqa: BLE001
                raise WordCOMUnavailableError("La integración COM de Word no estuvo disponible.") from exc
        raise WordCOMUnavailableError("La integración COM de Word no estuvo disponible.")

    @staticmethod
    def _normalize_text(text: Any) -> str:
        normalized = str(text or "").replace("\r", "\n")
        while "\n\n\n" in normalized:
            normalized = normalized.replace("\n\n\n", "\n\n")
        return normalized.strip()

    @staticmethod
    def _document_window_title(document) -> str | None:
        name = getattr(document, "Name", None)
        if name:
            return f"{name} - Word"
        return "Word"

    @staticmethod
    def _ends_with_document_marker(document) -> bool:
        try:
            return str(document.Range(max(int(document.Content.End) - 1, 0), int(document.Content.End)).Text) == "\r"
        except Exception:  # noqa: BLE001
            return True
