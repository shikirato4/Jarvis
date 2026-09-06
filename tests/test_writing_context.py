from jarvis.writing_runtime.models import WritingContinuationRequest, WritingContext
from jarvis.writing_runtime.style import WritingStyleAnalyzer


class _NullCharactersModel:
    def infer_json(self, **kwargs):
        return {"text_type": "casual", "characters": None, "confidence": 0.8}


def test_writing_style_analyzer_normalizes_null_characters() -> None:
    profile = WritingStyleAnalyzer(_NullCharactersModel()).analyze(
        WritingContext(combined_context="Texto breve", language="es"),
        correlation_id="regression-null-characters",
    )
    assert profile.characters == []
    assert profile.confidence == 0.8


def test_writing_context_detects_word_and_recent_text(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="writing context")
    jarvis_app.runtime_service.ui_focus_window({"target": "Word"})
    backend = jarvis_app.ui_automation_service._backend  # noqa: SLF001
    backend.typed_text = "Capítulo 3. Elena abrió la puerta y sostuvo la respiración mientras escuchaba pasos en el pasillo."
    analysis = jarvis_app.runtime_service.writing_analyze(
        WritingContinuationRequest(
            prompt="continúa mi libro",
            target_window="Word",
            write_directly=False,
        )
    )
    assert analysis.context.application_name == "word"
    assert "Elena" in analysis.context.combined_context
    assert analysis.style_profile.text_type.value in {"story", "casual", "technical"}
