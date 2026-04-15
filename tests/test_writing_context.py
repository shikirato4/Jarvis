from jarvis.writing_runtime.models import WritingContinuationRequest


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
