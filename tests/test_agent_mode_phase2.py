from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.desktop_agent_runtime import (
    AgentAction,
    AgentMode,
    AgentPermissionMode,
    AgentSafetyDecision,
    AgentSafetyGate,
    AgentSkill,
    AgentSkillRegistry,
    AgentTaskQueue,
    DesktopAgentDryRunPlanner,
    GitHubRepoSearchSkill,
    RepoRanker,
    RollbackPlanner,
    SafeLearningFilter,
)
from jarvis.desktop_agent_runtime.memory import DesktopAgentMemoryManager
from jarvis.desktop_agent_runtime.models import DesktopAgentPhase, DesktopAgentStep, DesktopStepActionType
from jarvis.desktop_agent_runtime.observer import DesktopAgentObserver
from jarvis.desktop_agent_runtime.observations import observation_summary_from_world
from jarvis.desktop_agent_runtime.world_model import DesktopWorldModelBuilder
from jarvis.desktop_runtime.intent_router import DesktopIntentRouter


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
        system_known_locations={
            "documents": str(tmp_path / "Documents"),
            "downloads": str(tmp_path / "Downloads"),
            "desktop": str(tmp_path / "Desktop"),
        },
    )


def test_agent_permission_modes_gate_actions() -> None:
    gate = AgentSafetyGate()

    observe = gate.authorize(
        AgentAction(action_type="observe_screen"),
        mode=AgentMode.GUIDED_CONTROL,
        permission_mode=AgentPermissionMode.LOCKDOWN,
    )
    safe_write = gate.authorize(
        AgentAction(action_type="create_folder", title="Crear carpeta"),
        mode=AgentMode.GUIDED_CONTROL,
        permission_mode=AgentPermissionMode.SAFE,
        confirmed=True,
    )
    normal_write = gate.authorize(
        AgentAction(action_type="create_folder", title="Crear carpeta"),
        mode=AgentMode.GUIDED_CONTROL,
        permission_mode=AgentPermissionMode.NORMAL,
    )
    pro_high = gate.authorize(
        AgentAction(action_type="run_script", title="instalar dependencia"),
        mode=AgentMode.GUIDED_CONTROL,
        permission_mode=AgentPermissionMode.PRO,
        strong_confirmed=True,
        pin_verified=True,
    )
    blocked = gate.authorize(
        AgentAction(action_type="read_secret", title="leer token de .env"),
        mode=AgentMode.GUIDED_CONTROL,
        permission_mode=AgentPermissionMode.PRO,
        strong_confirmed=True,
        pin_verified=True,
    )

    assert observe.decision == AgentSafetyDecision.BLOCK
    assert safe_write.decision == AgentSafetyDecision.BLOCK
    assert normal_write.decision == AgentSafetyDecision.REQUIRE_CONFIRMATION
    assert pro_high.decision == AgentSafetyDecision.ALLOW
    assert blocked.decision == AgentSafetyDecision.BLOCK


def test_desktop_agent_dry_run_does_not_create_folder(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    target = tmp_path / "Documents" / "dry run fase 2"

    result = DesktopAgentDryRunPlanner(settings=settings).plan(
        {"goal": 'crea una carpeta nueva en la carpeta de documentos llamada "dry run fase 2"'}
    ).to_dict()

    assert result["status"] == "dry_run"
    assert result["executed"] is False
    assert result["modifies_system"] is False
    assert result["steps"][0]["requires_confirmation"] is True
    assert result["steps"][0]["rollback"]["rollback_available"] is True
    assert not target.exists()


def test_runtime_dry_run_and_safe_mode_do_not_modify_files(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    target = tmp_path / "Documents" / "safe mode block"
    try:
        dry = app.runtime_service.desktop_agent_dry_run(
            {"goal": 'crea una carpeta nueva en la carpeta de documentos llamada "safe mode block"'}
        )
        assert dry["status"] == "dry_run"
        assert not target.exists()

        app.runtime_service.desktop_agent_set_permission_mode("safe")
        receipt = app.runtime_service.desktop_agent_run(
            {
                "goal": 'crea una carpeta nueva en la carpeta de documentos llamada "safe mode block"',
                "metadata": {"confirmed": True},
            }
        )
        assert receipt.status == DesktopAgentPhase.BLOCKED
        assert not target.exists()
    finally:
        app.stop()


def test_desktop_chat_router_sends_simulation_to_dry_run_not_execution() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.dry_run_requests: list[dict] = []
            self.run_requests: list[dict] = []

        def desktop_agent_dry_run(self, request: dict) -> dict:
            self.dry_run_requests.append(request)
            return {"status": "dry_run", "summary": "simulacion lista", "steps": []}

        def desktop_agent_run(self, request: dict):  # pragma: no cover - should never be reached
            self.run_requests.append(request)
            raise AssertionError("dry run prompt should not execute desktop_agent_run")

    runtime = Runtime()
    router = DesktopIntentRouter(SimpleNamespace(runtime=runtime), action_executor=None)
    decision = router.classify('simula crear una carpeta en documentos llamada "x"')
    result, summary, route = router.execute(decision)

    assert decision.category == "desktop_agent_dry_run"
    assert route == "desktop_agent_runtime.dry_run"
    assert result["status"] == "dry_run"
    assert summary == "simulacion lista"
    assert runtime.dry_run_requests[0]["metadata"]["dry_run"] is True
    assert runtime.run_requests == []


def test_sensitive_window_observation_degrades_without_crashing(tmp_path: Path) -> None:
    app = build_application(_settings(tmp_path))
    app.start()
    try:
        observer = DesktopAgentObserver(
            runtime=app.runtime_service,
            ui_backend=app.ui_automation_service._backend,  # noqa: SLF001
            memory=DesktopAgentMemoryManager(),
        )
        world = DesktopWorldModelBuilder().create({"goal": "observa mi pantalla"})
        original_describe = app.runtime_service.vision_describe_active_window
        original_awareness = app.runtime_service.vision_ui_awareness

        def _sensitive_window():
            raise RuntimeError("capture blocked for a sensitive window")

        def _sensitive_screen(_request):
            raise RuntimeError("screen capture blocked for a sensitive window")

        app.runtime_service.vision_describe_active_window = _sensitive_window  # type: ignore[method-assign]
        app.runtime_service.vision_ui_awareness = _sensitive_screen  # type: ignore[method-assign]
        try:
            observed = observer.observe(world, phase=DesktopAgentPhase.OBSERVING)
            summary = observation_summary_from_world(observed)
            assert observed.last_observation_summary
            assert summary.sensitive_blocked is True
            assert summary.status == "degraded"
        finally:
            app.runtime_service.vision_describe_active_window = original_describe  # type: ignore[method-assign]
            app.runtime_service.vision_ui_awareness = original_awareness  # type: ignore[method-assign]
    finally:
        app.stop()


def test_rollback_planner_reports_supported_and_unsupported_actions() -> None:
    planner = RollbackPlanner()
    create_folder = DesktopAgentStep(
        step_id="create-folder",
        title="Crear carpeta",
        action_type=DesktopStepActionType.CREATE_FOLDER,
        precondition="path resolved",
        action="crear carpeta",
        payload={"path": "C:/tmp/demo"},
    )
    click = DesktopAgentStep(
        step_id="click",
        title="Click",
        action_type=DesktopStepActionType.CLICK_TARGET,
        precondition="target visible",
        action="click",
        payload={"label": "OK"},
    )

    assert planner.for_step(create_folder).rollback_available is True
    assert planner.for_step(click).rollback_available is False


def test_agent_task_queue_add_cancel_continue() -> None:
    queue = AgentTaskQueue()
    first = queue.add("aprender patrones PySide6", task_type="learning", priority=1, requires_confirmation=True)
    second = queue.add("revisar docs", task_type="research", priority=5)

    assert queue.next_pending() == first
    assert [item.id for item in queue.list()] == [first.id, second.id]
    cancelled = queue.cancel(first.id)
    assert cancelled.status.value == "cancelled"
    assert queue.next_pending() == second


def test_agent_skills_do_not_execute_without_runtime_executor() -> None:
    registry = AgentSkillRegistry([AgentSkill("inspect_screen", "Inspectar", "eyes", "low", action_type="inspect_screen")])
    with pytest.raises(ValueError):
        registry.register(AgentSkill("inspect_screen", "Duplicado", "eyes", "low"))

    skill = registry.get("inspect_screen")
    assert skill is not None
    assert skill.dry_run({})["would_execute"] is False
    assert skill.execute({})["status"] == "blocked"
    allowed = registry.authorize("inspect_screen", {}, permission_mode=AgentPermissionMode.NORMAL)
    assert allowed["decision"] == "allow"


def test_github_learning_search_blocks_secrets_and_ranks_without_cloning() -> None:
    calls: list[dict] = []

    class Discovery:
        def search(self, query: str, *, max_results: int, language: str | None = None, topic: str | None = None) -> dict:
            calls.append({"query": query, "max_results": max_results, "language": language, "topic": topic})
            return {
                "status": "ok",
                "query": query,
                "results": [
                    {
                        "full_name": "owner/active-agent",
                        "description": "Python desktop agent with PySide6 tests",
                        "language": "Python",
                        "topics": ["pyside6", "agent"],
                        "stargazers_count": 4200,
                        "forks_count": 200,
                        "license": "mit",
                        "archived": False,
                        "fork": False,
                    },
                    {
                        "full_name": "owner/old-agent",
                        "description": "agent",
                        "language": "Python",
                        "license": "unknown",
                        "archived": True,
                        "fork": True,
                    },
                ],
            }

    skill = GitHubRepoSearchSkill(Discovery())
    blocked = skill.search("aprende de este token abc123")
    result = skill.search("PySide6 desktop agent", max_results=5)

    assert blocked["status"] == "blocked"
    assert calls and calls[0]["query"] == "PySide6 desktop agent"
    assert result["results"][0]["full_name"] == "owner/active-agent"
    assert result["message"].endswith("confirmacion explicita.")
    assert any("unknown license" in warning for warning in result["results"][1]["warnings"])


def test_safe_learning_filter_redacts_secrets_and_unsafe_commands() -> None:
    from jarvis.desktop_agent_runtime.learning import LearningArtifact

    artifact = LearningArtifact(
        title="Patron",
        source_url="https://github.com/example/repo",
        source_type="github",
        summary="usa API key secreta",
        patterns=["guardar token en .env"],
        safe_commands=["python -m pytest", "rm -rf ."],
        reusable_context="password=abc",
    )
    sanitized = SafeLearningFilter().sanitize_artifact(artifact)

    assert sanitized.summary == "[redacted]"
    assert sanitized.reusable_context == "[redacted]"
    assert "python -m pytest" in sanitized.safe_commands
    assert "rm -rf ." not in sanitized.safe_commands
    assert sanitized.blocked_sections


def test_repo_ranker_warns_for_risky_repositories() -> None:
    scored = RepoRanker().score(
        {
            "full_name": "owner/risky",
            "description": "agent",
            "private": True,
            "archived": True,
            "fork": True,
            "license": "unknown",
            "size": 999999,
        },
        "agent",
    )

    assert scored["security_risk"] == "high"
    assert any("private repository blocked" in warning for warning in scored["warnings"])
    assert any("archived repository" in warning for warning in scored["warnings"])
