from __future__ import annotations

from types import SimpleNamespace

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.core.errors import WritingRuntimeError
from jarvis.desktop import build_desktop_runtime
from jarvis.integrations.word_com import WordCOMBackend, WordCOMError
from jarvis.writing_runtime.models import WritingContinuationRequest


class FakeWordRange:
    def __init__(self, document: FakeWordDocument, start: int, end: int) -> None:
        self._document = document
        self._start = start
        self._end = end

    @property
    def Text(self) -> str:
        return self._document._text[self._start : self._end]

    @Text.setter
    def Text(self, value: str) -> None:
        self._document._text = self._document._text[: self._start] + value + self._document._text[self._end :]


class FakeWordContent:
    def __init__(self, document: FakeWordDocument) -> None:
        self._document = document

    @property
    def Text(self) -> str:
        return self._document._text

    @property
    def End(self) -> int:
        return len(self._document._text)


class FakeWordDocument:
    def __init__(self, text: str, *, name: str = "Historia.docx") -> None:
        self._text = text if text.endswith("\r") else f"{text}\r"
        self.Name = name
        self.FullName = f"C:/docs/{name}"
        self.Saved = True
        self.save_calls = 0

    @property
    def Content(self) -> FakeWordContent:
        return FakeWordContent(self)

    def Range(self, start: int, end: int) -> FakeWordRange:
        return FakeWordRange(self, start, end)

    def Save(self) -> None:
        self.save_calls += 1
        self.Saved = True


class FakeWordApplication:
    def __init__(self, document: FakeWordDocument | None, *, selection_end: int | None = None) -> None:
        self.ActiveDocument = document
        end = selection_end if selection_end is not None else (len(document._text) - 1 if document is not None else 0)
        self.Selection = SimpleNamespace(Range=SimpleNamespace(End=end))


class FailingWordBackend:
    def __init__(self, message: str = "No pude usar la integración nativa de Word; probando ruta alternativa.") -> None:
        self.message = message

    def detect_if_word_target(self, **kwargs) -> bool:  # noqa: ARG002
        return True

    def read_active_document_text(self) -> str:
        raise WordCOMError(self.message)

    def get_document_metadata(self) -> dict[str, str]:
        raise WordCOMError(self.message)

    def insert_text_at_cursor(self, text: str) -> dict[str, str]:  # noqa: ARG002
        raise WordCOMError(self.message)


def _settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
    )


def _install_word_backend(app, backend) -> None:
    app.writing_runtime_service._context_resolver._word_backend = backend  # noqa: SLF001
    app.writing_runtime_service._editor._word_backend = backend  # noqa: SLF001


def test_word_backend_detects_word_targets() -> None:
    backend = WordCOMBackend(app_provider=lambda: None)
    assert backend.detect_if_word_target(window_title="Mi novela - Word") is True
    assert backend.detect_if_word_target(process_name="WINWORD.EXE") is True
    assert backend.detect_if_word_target(target_hint="Microsoft Word") is True
    assert backend.detect_if_word_target(window_title="Notepad") is False


def test_word_backend_reads_active_document_text() -> None:
    document = FakeWordDocument("Capítulo 7. Elena abrió la carta.")
    backend = WordCOMBackend(app_provider=lambda: FakeWordApplication(document))

    assert backend.is_word_available() is True
    assert "Elena abrió la carta." in backend.read_active_document_text()
    metadata = backend.get_document_metadata()
    assert metadata["name"] == "Historia.docx"
    assert metadata["window_title"] == "Historia.docx - Word"


def test_word_backend_inserts_text_at_cursor_and_saves() -> None:
    document = FakeWordDocument("Hola mundo")
    backend = WordCOMBackend(app_provider=lambda: FakeWordApplication(document, selection_end=4))

    backend.insert_text_at_cursor(" brillante")
    backend.save_document()

    assert document._text == "Hola brillante mundo\r"
    assert document.save_calls == 1


def test_writing_continue_uses_word_com_without_focus_window(tmp_path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    document = FakeWordDocument("Capítulo inicial con suficiente contexto narrativo para continuar. " * 3)
    _install_word_backend(app, WordCOMBackend(app_provider=lambda: FakeWordApplication(document)))
    app.ui_automation_service.focus_window = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("focus_window no debe usarse para Word"))  # noqa: ARG005
    app.ui_automation_service.write_text = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("write_text UI no debe usarse para Word"))  # noqa: ARG005
    backend = app.ui_automation_service._backend  # noqa: SLF001
    backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
    try:
        response = desktop.send_chat("continua mi libro")
        assert "he continuado el texto en word" in response.message.content.casefold()
        assert "brillante" not in response.message.content.casefold()
        assert "continua mi libro" not in document._text.casefold()
        assert "Capítulo inicial" in document._text
    finally:
        app.stop()


def test_writing_continue_falls_back_to_ui_when_word_com_fails(tmp_path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        app.runtime_service.switch_mode("operator", reason="word com fallback")
        _install_word_backend(app, FailingWordBackend())
        backend = app.ui_automation_service._backend  # noqa: SLF001
        backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Mi libro"})  # noqa: SLF001
        backend.typed_text = "Contexto suficiente para seguir escribiendo en modo fallback. " * 3

        receipt = app.runtime_service.writing_continue(
            WritingContinuationRequest(prompt="continua mi libro", target_window="Word", desired_words=60, write_directly=True)
        )

        assert receipt.success is True
        assert receipt.written_text
        assert receipt.written_text in backend.typed_text
    finally:
        app.stop()


def test_writing_continue_avoids_operator_mode_and_ui_deadline_when_word_com_available(tmp_path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    document = FakeWordDocument("Historia con contexto suficiente para continuar en Word por COM. " * 3)
    _install_word_backend(app, WordCOMBackend(app_provider=lambda: FakeWordApplication(document)))
    app.ui_automation_service.write_text = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("operation deadline exceeded"))  # noqa: ARG005
    backend = app.ui_automation_service._backend  # noqa: SLF001
    backend._active_window = backend.list_windows()[1].model_copy(update={"title": "Word - Historia abierta"})  # noqa: SLF001
    try:
        analysis = app.runtime_service.writing_analyze(
            WritingContinuationRequest(prompt="ves la historia que tengo abierta en Word?", target_window="Word", write_directly=False)
        )
        assert analysis.context.metadata["integration"] == "word_com"
        receipt = app.runtime_service.writing_continue(
            WritingContinuationRequest(prompt="continua mi libro", target_window="Word", desired_words=60, write_directly=True)
        )
        assert receipt.success is True
        assert receipt.verification_summary["ui"]["data"]["integration"] == "word_com"
    finally:
        app.stop()


def test_word_com_no_active_document_returns_clear_error(tmp_path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    _install_word_backend(app, WordCOMBackend(app_provider=lambda: FakeWordApplication(None)))
    try:
        try:
            app.runtime_service.writing_analyze(
                WritingContinuationRequest(prompt="ves la historia que tengo abierta en Word?", target_window="Word", write_directly=False)
            )
        except WritingRuntimeError as exc:
            assert "Word no tiene un documento activo." in str(exc)
        else:
            raise AssertionError("expected WritingRuntimeError")
    finally:
        app.stop()
