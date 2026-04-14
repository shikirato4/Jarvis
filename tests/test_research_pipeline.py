from pathlib import Path

from jarvis.research_runtime.models import ResearchRunRequest, SimulatedResearchSource


def test_research_pipeline_generates_report(jarvis_app, tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Jarvis runtime is modular and uses separate services.\nMission control exists for approvals.\n", encoding="utf-8")
    task = jarvis_app.runtime_service.research_run(
        ResearchRunRequest(
            query="Investiga la arquitectura del runtime de Jarvis",
            paths=[str(tmp_path / "notes.md")],
            source_scope=("workspace", "simulated"),
            simulated_sources=[
                SimulatedResearchSource(
                    title="spec",
                    content="Jarvis has autonomy, mission control and modular runtimes.",
                    location="sim://spec",
                )
            ],
        )
    )
    assert task.status.value == "completed"
    assert task.report is not None
    assert task.report.short_summary
    assert task.findings
    assert task.validations


def test_research_can_delegate_to_autonomy(jarvis_app) -> None:
    task = jarvis_app.runtime_service.research_run(
        {
            "query": "Research the runtime architecture",
            "run_via_autonomy": True,
            "source_scope": ["simulated"],
            "simulated_sources": [{"title": "spec", "content": "Jarvis runtime is modular.", "location": "sim://spec"}],
        }
    )
    assert task.mission_id is not None
    assert task.status.value == "delegated"
