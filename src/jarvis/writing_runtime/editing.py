from __future__ import annotations

from datetime import datetime, timezone

from jarvis.integrations import WordCOMBackend, WordCOMError, WordCOMUnavailableError, WordNoActiveDocumentError
from jarvis.ui_automation.base import UIOperationReceipt, UIOperationStatus, UIRiskLevel
from jarvis.ui_automation.base import WriteTextRequest

from .models import WritingContext, WritingContinuationRequest


class WritingEditor:
    def __init__(self, ui_automation, settings=None, word_backend: WordCOMBackend | None = None, logger=None) -> None:
        self._ui = ui_automation
        self._settings = settings
        self._word_backend = word_backend
        self._logger = logger

    def write(self, text: str, context: WritingContext, request: WritingContinuationRequest, *, correlation_id: str):
        if self._should_use_word_com(context=context, request=request):
            receipt = self._write_via_word_com(text=text, context=context, request=request, correlation_id=correlation_id)
            if receipt is not None:
                return receipt
        return self._ui.write_text(
            WriteTextRequest(
                text=text,
                mode="copilot",
                block_size=getattr(self._settings, "writing_ui_block_size", None),
                typing_interval_ms=getattr(self._settings, "writing_ui_typing_interval_ms", None),
                pause_between_blocks_ms=getattr(self._settings, "writing_ui_pause_between_blocks_ms", None),
                focus_target=request.target_window or context.window_title,
                ensure_window_contains=request.ensure_window_contains or request.target_window or context.window_title,
                approved=bool(request.metadata.get("approved", False)),
                timeout_ms=getattr(self._settings, "writing_ui_write_timeout_ms", None),
                watchdog_timeout_ms=getattr(self._settings, "writing_ui_write_timeout_ms", None),
            ),
            correlation_id=correlation_id,
        )

    def _write_via_word_com(self, *, text: str, context: WritingContext, request: WritingContinuationRequest, correlation_id: str) -> UIOperationReceipt | None:
        if self._word_backend is None:
            return None
        started_at = datetime.now(timezone.utc)
        try:
            write_result = self._word_backend.insert_text_at_cursor(text)
            if bool(request.metadata.get("save_document", False)):
                self._word_backend.save_document()
                write_result["saved"] = True
            return UIOperationReceipt(
                correlation_id=correlation_id,
                operation_name="word_com.write_text",
                risk_level=UIRiskLevel.MEDIUM,
                success=True,
                status=UIOperationStatus.EXECUTED,
                message="He continuado el texto en Word.",
                confirmation_required=False,
                security_decision="allowed",
                data={
                    "integration": "word_com",
                    "target_window": context.window_title or request.target_window or "Word",
                    **write_result,
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        except WordNoActiveDocumentError:
            raise
        except WordCOMUnavailableError:
            self._log_word_fallback("La integración COM de Word no estuvo disponible.")
            return None
        except WordCOMError as exc:
            self._log_word_fallback(str(exc))
            return None

    def _should_use_word_com(self, *, context: WritingContext, request: WritingContinuationRequest) -> bool:
        if self._word_backend is None:
            return False
        return self._word_backend.detect_if_word_target(
            window_title=context.window_title,
            target_hint=request.target_window or request.ensure_window_contains,
        ) or context.metadata.get("integration") == "word_com"

    def _log_word_fallback(self, reason: str) -> None:
        if self._logger is None:
            return
        self._logger.warning("word_com_write_fallback_to_ui", extra={"reason": reason})
