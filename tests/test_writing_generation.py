from jarvis.writing_runtime.models import WritingContinuationRequest
from jarvis.writing_runtime.generation import WritingGenerator


def test_writing_generation_preserves_context_without_model_provider(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="writing generation")
    jarvis_app.runtime_service.ui_focus_window({"target": "Word"})
    backend = jarvis_app.ui_automation_service._backend  # noqa: SLF001
    backend.typed_text = "La arquitectura del sistema sigue un diseño modular con servicios separados y contratos estables."
    receipt = jarvis_app.runtime_service.writing_continue(
        WritingContinuationRequest(
            prompt="continúa con el mismo tono",
            target_window="Word",
            write_directly=False,
            desired_words=80,
        )
    )
    assert receipt.success is True
    assert receipt.generated_text
    assert len(receipt.generated_text.split()) >= 20


def test_writing_generation_normalizes_qualitative_model_confidence() -> None:
    assert WritingGenerator._normalize_confidence("moderate") == 0.6
    assert WritingGenerator._normalize_confidence("VERY_HIGH") == 0.95
    assert WritingGenerator._normalize_confidence("unexpected") == 0.7
    assert WritingGenerator._normalize_confidence(4.2) == 1.0
