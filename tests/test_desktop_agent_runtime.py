from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app as cli_app
from jarvis.config import Settings
from jarvis.desktop import build_desktop_runtime
from jarvis.desktop_agent_runtime.memory import DesktopAgentMemoryManager
from jarvis.desktop_agent_runtime.models import (
    DesktopAgentModelDecision,
    DesktopAgentModelSuggestion,
    DesktopAgentPhase,
    DesktopAgentRiskLevel,
    DesktopAgentStep,
    DesktopAgentVerificationResult,
    DesktopStepActionType,
    DesktopVerificationStatus,
)
from jarvis.desktop_agent_runtime.observer import DesktopAgentObserver
from jarvis.desktop_agent_runtime.planner import DesktopAgentPlanner
from jarvis.desktop_agent_runtime.policies import DesktopAgentPolicyEngine
from jarvis.desktop_agent_runtime.recovery import DesktopAgentRecoveryEngine
from jarvis.desktop_agent_runtime.verifier import DesktopAgentVerifier
from jarvis.desktop_agent_runtime.world_model import DesktopWorldModelBuilder


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embeddings_enabled=False,
        ui_backend_kind="in_memory",
        system_backend_kind="in_memory",
        system_search_roots=(tmp_path,),
    )


def test_desktop_agent_builds_world_state(tmp_path: Path) -> None:
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    assert world.current_goal == "abre chrome y busca youtube"
    assert world.phase == DesktopAgentPhase.PENDING
    assert world.mission_id
    assert world.target_application == "chrome"


def test_desktop_agent_planner_generates_multistep_browser_plan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    plan = DesktopAgentPlanner(settings).plan(world)
    assert [step.step_id for step in plan.steps] == [
        "open-browser",
        "focus-browser",
        "focus-address-bar",
        "type-query",
        "submit-query",
    ]


def test_desktop_agent_recovery_retries_write_step(tmp_path: Path) -> None:
    world = DesktopWorldModelBuilder().create({"goal": "ve a la ventana activa y escribe esto: hola"})
    step = DesktopAgentStep(
        step_id="write-active",
        title="Escribir texto",
        action_type=DesktopStepActionType.WRITE_TEXT,
        precondition="ready",
        action="write",
        payload={"text": "hola"},
        risk_level=DesktopAgentRiskLevel.LOW,
        max_retries=1,
    )
    verification = DesktopAgentVerificationResult(
        status=DesktopVerificationStatus.PARTIAL,
        note="Expected visible text containing 'hola'.",
        missing=["visible_text_contains:hola"],
    )
    recovered_world, decision = DesktopAgentRecoveryEngine(memory=DesktopAgentMemoryManager()).recover(world, step, verification)
    assert decision.should_retry is True
    assert recovered_world.memory.attempted_fallbacks
    assert decision.strategy == "reobserve_then_retry"


def test_desktop_agent_recovery_avoids_repeating_identical_attempts(tmp_path: Path) -> None:
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    step = DesktopAgentStep(
        step_id="focus-browser",
        title="Enfocar navegador",
        action_type=DesktopStepActionType.FOCUS_WINDOW,
        precondition="ready",
        action="focus",
        payload={"target_window": "chrome"},
        max_retries=2,
    )
    memory = DesktopAgentMemoryManager()
    engine = DesktopAgentRecoveryEngine(memory=memory)
    verification = DesktopAgentVerificationResult(
        status=DesktopVerificationStatus.FAILED,
        note="Expected active window containing 'chrome'.",
        missing=["active_window_contains:chrome"],
    )
    world, first = engine.recover(world, step, verification)
    assert first.strategy == "refocus_target_window"
    world, second = engine.recover(world, step, verification)
    assert second.should_replan is True
    assert second.strategy == "heuristic_replan_current_subgoal"


def test_desktop_agent_recovery_prefers_model_replan_when_available(tmp_path: Path) -> None:
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    step = DesktopAgentStep(
        step_id="type-query",
        title="Escribir busqueda",
        action_type=DesktopStepActionType.WRITE_TEXT,
        precondition="ready",
        action="write",
        payload={"text": "youtube", "target_window": "chrome"},
        max_retries=1,
    )
    verification = DesktopAgentVerificationResult(
        status=DesktopVerificationStatus.FAILED,
        note="Expected visible text containing 'youtube'.",
        missing=["visible_text_contains:youtube"],
    )
    suggestion = DesktopAgentModelSuggestion(
        decision=DesktopAgentModelDecision.REPLAN,
        strategy="model_browser_retry",
        rationale="The browser is active but the query field is not grounded; refocus and type again.",
        steps=[
            DesktopAgentStep(
                step_id="model-focus",
                title="Refocus browser",
                action_type=DesktopStepActionType.FOCUS_WINDOW,
                precondition="ready",
                action="focus browser",
                payload={"target_window": "chrome"},
            )
        ],
    )
    _, decision = DesktopAgentRecoveryEngine(memory=DesktopAgentMemoryManager()).recover(world, step, verification, model_suggestion=suggestion)
    assert decision.should_replan is True
    assert decision.strategy == "model_browser_retry"


def test_desktop_agent_planner_parses_model_replan(tmp_path: Path) -> None:
    class RuntimeStub:
        def infer_model(self, request):
            class Response:
                content = json.dumps(
                    {
                        "decision": "replan",
                        "strategy": "browser_refocus_then_search",
                        "rationale": "Chrome is available, but the search field was not visible.",
                        "steps": [
                            {
                                "title": "Refocus Chrome",
                                "action_type": "focus_window",
                                "action": "Focus Chrome",
                                "payload": {"target_window": "chrome"},
                                "verification": {"active_window_contains": "chrome"},
                            },
                            {
                                "title": "Type query again",
                                "action_type": "write_text",
                                "action": "Write youtube",
                                "payload": {"text": "youtube", "target_window": "chrome"},
                                "verification": {"visible_text_contains": ["youtube"]},
                            },
                        ],
                        "metadata": {"confidence": "high"},
                    }
                )

            return Response()

    planner = DesktopAgentPlanner(_settings(tmp_path), runtime=RuntimeStub())
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    world.current_subgoal = "colocar la consulta en el campo visible"
    failed_step = DesktopAgentStep(
        step_id="type-query",
        title="Escribir busqueda",
        action_type=DesktopStepActionType.WRITE_TEXT,
        precondition="ready",
        action="write",
        payload={"text": "youtube", "target_window": "chrome"},
    )
    verification = DesktopAgentVerificationResult(
        status=DesktopVerificationStatus.FAILED,
        note="Expected visible text containing 'youtube'.",
        missing=["visible_text_contains:youtube"],
    )
    plan = planner.replan(world, reason=verification.note, failed_step=failed_step, verification=verification)
    assert plan.metadata["source"] == "model"
    assert plan.strategy == "browser_refocus_then_search"
    assert [step.action_type for step in plan.steps] == [DesktopStepActionType.FOCUS_WINDOW, DesktopStepActionType.WRITE_TEXT]


def test_desktop_agent_policy_blocks_blocklisted_hotkey(tmp_path: Path) -> None:
    engine = DesktopAgentPolicyEngine(_settings(tmp_path))
    step = DesktopAgentStep(
        step_id="danger",
        title="Danger",
        action_type=DesktopStepActionType.HOTKEY,
        precondition="ready",
        action="alt+f4",
        payload={"keys": ("alt", "f4")},
        risk_level=DesktopAgentRiskLevel.HIGH,
    )
    result = engine.assess_step(step)
    assert result.decision.value == "deny"


def test_desktop_agent_run_executes_and_verifies_browser_search(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        receipt = app.runtime_service.desktop_agent_run({"goal": "abre chrome y busca youtube"})
        backend = app.ui_automation_service._backend  # noqa: SLF001
        assert receipt.success is True
        assert receipt.status == DesktopAgentPhase.COMPLETED
        assert backend.hotkeys[-2] == ("ctrl", "l")
        assert backend.hotkeys[-1] == ("enter",)
        assert "youtube" in backend.typed_text
        assert receipt.step_receipts
        assert receipt.world_state.context_signals
        assert receipt.mission_snapshot["current_subgoal"] == "ejecutar la busqueda y mantener evidencia de contexto"
        assert receipt.subtasks
        assert receipt.checkpoints
        assert receipt.progress.total_steps >= 5
    finally:
        app.stop()


def test_desktop_agent_run_uses_model_replan_after_failure(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        original_verify = app.desktop_agent_runtime_service._verifier.verify  # noqa: SLF001
        failed_once = {"done": False}

        def _verify(world, step, action_result):
            if step.step_id == "type-query" and not failed_once["done"]:
                failed_once["done"] = True
                return DesktopAgentVerificationResult(
                    status=DesktopVerificationStatus.FAILED,
                    note="Expected visible text containing 'youtube'.",
                    missing=["visible_text_contains:youtube"],
                    observed={"visible_text": world.visible_text},
                )
            return original_verify(world, step, action_result)

        def _infer_model(request):
            class Response:
                content = json.dumps(
                    {
                        "decision": "replan",
                        "strategy": "model_refocus_and_retry_query",
                        "rationale": "The query was not grounded after typing; refocus Chrome and type again.",
                        "steps": [
                            {
                                "title": "Refocus Chrome",
                                "action_type": "focus_window",
                                "action": "Refocus Chrome",
                                "payload": {"target_window": "chrome"},
                                "verification": {"active_window_contains": "chrome"},
                            },
                            {
                                "title": "Type youtube again",
                                "action_type": "write_text",
                                "action": "Write youtube again",
                                "payload": {"text": "youtube", "target_window": "chrome"},
                                "verification": {"visible_text_contains": ["youtube"]},
                            },
                            {
                                "title": "Submit search",
                                "action_type": "hotkey",
                                "action": "Press enter",
                                "payload": {"keys": ["enter"], "target_window": "chrome"},
                                "verification": {"active_window_contains": "chrome", "visible_text_contains": ["youtube"]},
                            },
                        ],
                        "metadata": {"confidence": "high"},
                    }
                )

            return Response()

        app.desktop_agent_runtime_service._verifier.verify = _verify  # type: ignore[method-assign]  # noqa: SLF001
        app.runtime_service.infer_model = _infer_model  # type: ignore[method-assign]

        receipt = app.runtime_service.desktop_agent_run({"goal": "abre chrome y busca youtube"})
        assert receipt.success is True
        assert receipt.plan is not None
        assert receipt.plan.metadata["source"] == "model"
        assert receipt.plan.strategy == "model_refocus_and_retry_query"
        assert any(item.recovery_strategy == "model_refocus_and_retry_query" for item in receipt.step_receipts)
    finally:
        app.stop()


def test_desktop_agent_run_handles_file_search_and_open(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("jarvis desktop agent", encoding="utf-8")
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        receipt = app.runtime_service.desktop_agent_run({"goal": "busca este archivo notes y abrelo"})
        assert receipt.success is True
        assert "search-file" in receipt.completed_steps
        assert "open-file" in receipt.completed_steps
        assert receipt.world_state.memory.completed_steps == receipt.completed_steps
    finally:
        app.stop()


def test_desktop_agent_observer_collects_grounded_context(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = build_application(settings)
    app.start()
    try:
        world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
        observer = DesktopAgentObserver(
            runtime=app.runtime_service,
            ui_backend=app.ui_automation_service._backend,  # noqa: SLF001
            memory=DesktopAgentMemoryManager(),
        )
        observed = observer.observe(world, phase=DesktopAgentPhase.OBSERVING)
        assert observed.active_window is not None
        assert observed.last_observation_summary
        assert any(signal.startswith("window_title:") for signal in observed.context_signals)
        assert "selection_available" not in observed.context_signals or observed.selection_text is not None
    finally:
        app.stop()


def test_desktop_agent_verifier_reports_partial_with_expected_and_observed(tmp_path: Path) -> None:
    world = DesktopWorldModelBuilder().create({"goal": "abre chrome y busca youtube"})
    world.visible_text = "youtube"
    world.context_signals = ["browser_active"]
    step = DesktopAgentStep(
        step_id="submit-query",
        title="Enviar",
        action_type=DesktopStepActionType.HOTKEY,
        precondition="ready",
        action="enter",
        payload={"keys": ("enter",)},
        verification={
            "active_window_contains": "chrome",
            "visible_text_contains": ["youtube"],
            "required_context_signals": ["browser_active"],
        },
    )
    result = DesktopAgentVerifier().verify(world, step, {"success": True})
    assert result.status == DesktopVerificationStatus.PARTIAL
    assert "active_window_contains:chrome" in result.missing
    assert result.observed["visible_text"] == "youtube"


def test_desktop_agent_runtime_status_exposes_latest_goal_and_observation(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        app.runtime_service.desktop_agent_run({"goal": "abre chrome y busca youtube"})
        status = app.runtime_service.desktop_agent_status()
        assert status["latest_goal"] == "abre chrome y busca youtube"
        assert status["latest_observation"]
        assert status["latest_step"]
    finally:
        app.stop()


def test_desktop_chat_routes_goal_to_desktop_agent(tmp_path: Path) -> None:
    app, desktop = build_desktop_runtime(_settings(tmp_path))
    try:
        response = desktop.send_chat("abre chrome y busca youtube")
        assert "Navegando" in response.message.content
        assert response.raw_result.get("status") == "completed"
        assert response.panel_snapshot is not None
        assert response.panel_snapshot.missions
    finally:
        app.stop()


def test_desktop_agent_api_route(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        response = client.post("/desktop-agent/run", json={"goal": "abre chrome y busca youtube"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["status"] == "completed"


def test_desktop_agent_cli_run(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(_settings(tmp_path)))
    result = runner.invoke(cli_app, ["desktop-agent", "run", "abre chrome y busca youtube"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["status"] == "completed"


def test_desktop_agent_pause_resume_abort_and_list(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        import time

        original_execute = app.desktop_agent_runtime_service._executor.execute  # noqa: SLF001

        def _slow_execute(world, step):
            time.sleep(0.05)
            return original_execute(world, step)

        app.desktop_agent_runtime_service._executor.execute = _slow_execute  # type: ignore[method-assign]  # noqa: SLF001
        mission = app.runtime_service.desktop_agent_run(
            {"goal": "abre chrome y busca youtube", "wait_for_completion": False}
        )
        mission_id = mission.mission_id
        paused = app.runtime_service.desktop_agent_pause(mission_id)
        assert paused.status == DesktopAgentPhase.PAUSED
        listed = app.runtime_service.desktop_agent_list()
        assert any(item.mission_id == mission_id for item in listed)
        resumed = app.runtime_service.desktop_agent_resume(mission_id)
        assert resumed.resume_count >= 1
        aborted = app.runtime_service.desktop_agent_abort(mission_id)
        assert aborted.status == DesktopAgentPhase.ABORTED
        assert aborted.abort_reason
    finally:
        app.stop()


def test_desktop_agent_resume_from_checkpoint(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        original_execute = app.desktop_agent_runtime_service._executor.execute  # noqa: SLF001
        paused_once = {"done": False}

        def _execute(world, step):
            result = original_execute(world, step)
            if step.step_id == "focus-browser" and not paused_once["done"]:
                paused_once["done"] = True
                app.desktop_agent_runtime_service.pause_mission(world.mission_id)
            return result

        app.desktop_agent_runtime_service._executor.execute = _execute  # type: ignore[method-assign]  # noqa: SLF001
        mission = app.runtime_service.desktop_agent_run({"goal": "abre chrome y busca youtube", "wait_for_completion": False})
        mission_id = mission.mission_id
        import time

        deadline = time.time() + 5
        while time.time() < deadline:
            status = app.runtime_service.desktop_agent_get(mission_id)
            if status.status == DesktopAgentPhase.PAUSED:
                break
            time.sleep(0.05)
        paused = app.runtime_service.desktop_agent_get(mission_id)
        assert paused.status == DesktopAgentPhase.PAUSED
        assert paused.checkpoints
        app.runtime_service.desktop_agent_resume(mission_id)
        deadline = time.time() + 5
        while time.time() < deadline:
            status = app.runtime_service.desktop_agent_get(mission_id)
            if status.status == DesktopAgentPhase.COMPLETED:
                break
            time.sleep(0.05)
        completed = app.runtime_service.desktop_agent_get(mission_id)
        assert completed.status == DesktopAgentPhase.COMPLETED
        assert completed.resume_count >= 1
        assert completed.progress.percent_complete == 100.0
    finally:
        app.stop()


def test_desktop_agent_api_pause_resume_abort_and_list(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    test_app = build_application(settings)
    import jarvis.api.app as api_module

    monkeypatch.setattr(api_module, "build_application", lambda: test_app)
    with TestClient(api_module.create_api_app()) as client:
        create = client.post("/desktop-agent/run", json={"goal": "abre chrome y busca youtube", "wait_for_completion": False})
        assert create.status_code == 200
        mission_id = create.json()["mission_id"]
        assert client.get(f"/desktop-agent/status/{mission_id}").status_code == 200
        assert client.get("/desktop-agent/list").status_code == 200
        assert client.post(f"/desktop-agent/pause/{mission_id}").status_code == 200
        assert client.post(f"/desktop-agent/resume/{mission_id}").status_code == 200
        abort = client.post(f"/desktop-agent/abort/{mission_id}")
        assert abort.status_code == 200
        assert abort.json()["status"] == "aborted"


def test_desktop_agent_cli_pause_resume_abort_and_list(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", lambda: build_application(_settings(tmp_path)))
    create = runner.invoke(cli_app, ["desktop-agent", "run", "abre chrome y busca youtube", "--detach"])
    assert create.exit_code == 0
    mission_id = json.loads(create.stdout)["mission_id"]
    status = runner.invoke(cli_app, ["desktop-agent", "status", "--mission", mission_id])
    assert status.exit_code == 0
    listed = runner.invoke(cli_app, ["desktop-agent", "list"])
    assert listed.exit_code == 0
    paused = runner.invoke(cli_app, ["desktop-agent", "pause", "--mission", mission_id])
    assert paused.exit_code == 0
    resumed = runner.invoke(cli_app, ["desktop-agent", "resume", "--mission", mission_id])
    assert resumed.exit_code == 0
    aborted = runner.invoke(cli_app, ["desktop-agent", "abort", "--mission", mission_id])
    assert aborted.exit_code == 0
    assert json.loads(aborted.stdout)["status"] == "aborted"
