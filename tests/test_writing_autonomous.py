from jarvis.writing_runtime.models import WritingContinuationRequest, WritingMode


def test_writing_autonomous_start_and_stop_delegate_to_autonomy(jarvis_app) -> None:
    jarvis_app.runtime_service.switch_mode("operator", reason="writing autonomous")
    receipt = jarvis_app.runtime_service.writing_autonomous_start(
        WritingContinuationRequest(
            prompt="avanza mientras no estoy",
            mode=WritingMode.AUTONOMOUS,
            target_window="Word",
            desired_words=180,
            write_directly=True,
        )
    )
    assert receipt.success is True
    assert receipt.task_id is not None
    task = jarvis_app.writing_runtime_service.get_task(receipt.task_id)
    assert task.mission_id is not None
    stop = jarvis_app.runtime_service.writing_autonomous_stop(receipt.task_id)
    assert stop.success is True
