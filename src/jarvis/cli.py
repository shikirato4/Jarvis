from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from typing import Any

import typer
import uvicorn

from jarvis.automation.service import AutomationDefinition
from jarvis.bootstrap import build_application
from jarvis.cognition.models import OrchestrationRequest
from jarvis.core.modes import ExecutionMode
from jarvis.desktop_agent_runtime import DesktopAgentMissionRequest
from jarvis.autonomy.base import MissionApprovalRequest, MissionControlActionRequest, MissionPlanRequest, MissionRequest
from jarvis.indexing_runtime.models import IndexRunRequest, IndexSourceCreateRequest, IndexSourceKind, IndexingTrigger
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.models_runtime.base import ModelRequest
from jarvis.research_runtime.models import ResearchBudget, ResearchRunRequest, SimulatedResearchSource
from jarvis.routing.models import TaskRequest
from jarvis.science_runtime import ScienceSimulationRequest, ScienceSolveRequest
from jarvis.security_runtime import SecurityAnalyzeRequest, SecurityPasswordCheckRequest
from jarvis.system_runtime.base import ResourceQuery, SystemOpenRequest, SystemResolveRequest, SystemSearchRequest
from jarvis.unity_runtime.base import (
    UnityAssetSearchRequest,
    UnityBridgeConnectRequest,
    UnityBridgeDisconnectRequest,
    UnityBridgeRequest,
    UnityEditorOperationRequest,
    UnityEditorOperationKind,
    UnityLaunchRequestModel,
    UnityProjectCreateRequest,
    UnityProjectQuery,
    UnityProjectResolveRequest,
    UnityScriptGenerationRequest,
    UnityScriptWriteRequest,
)
from jarvis.ui_automation.base import ClickRequest, FocusWindowRequest, MoveMouseRequest, ShortcutRequest, UIAutomationMode, WriteTextRequest
from jarvis.voice_runtime.base import VoiceSessionRequest
from jarvis.writing_runtime.models import WritingContinuationRequest, WritingMode

app = typer.Typer(help="Jarvis cognitive operating system CLI.")
semantic_app = typer.Typer(help="Semantic memory operations.")
ui_app = typer.Typer(help="Desktop automation and direct writing.")
voice_app = typer.Typer(help="Voice runtime, transcription, synthesis and dictation.")
autonomy_app = typer.Typer(help="Autonomous multi-step mission runtime.")
system_app = typer.Typer(help="System runtime for search, resolution and safe opening of system resources.")
unity_app = typer.Typer(help="Unity runtime for projects, assets, scripts and editor operations.")
ops_app = typer.Typer(help="Operational diagnostics, recovery and retention.")
research_app = typer.Typer(help="Deep research runtime for multi-step investigations.")
writing_app = typer.Typer(help="Writing copilot runtime for contextual continuation and autonomous writing.")
science_app = typer.Typer(help="Science runtime for symbolic solving and simulations.")
security_app = typer.Typer(help="Security runtime for password checks and local analysis.")
index_app = typer.Typer(help="Indexing runtime for workspace and document synchronization.")
desktop_agent_app = typer.Typer(help="Persistent desktop agent runtime with observe-plan-act-verify-recover loop.")
app.add_typer(semantic_app, name="semantic")
app.add_typer(ui_app, name="ui")
app.add_typer(voice_app, name="voice")
app.add_typer(autonomy_app, name="autonomy")
app.add_typer(system_app, name="system")
app.add_typer(unity_app, name="unity")
app.add_typer(ops_app, name="ops")
app.add_typer(research_app, name="research")
app.add_typer(writing_app, name="writing")
app.add_typer(science_app, name="science")
app.add_typer(security_app, name="security")
app.add_typer(index_app, name="index")
app.add_typer(desktop_agent_app, name="desktop-agent")


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        normalized = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)", r'\1"\2"\3', raw)
        normalized = normalized.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null")
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw)
            if not isinstance(parsed, dict):
                raise ValueError("payload must resolve to a dictionary")
            return parsed


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run("jarvis.api.app:create_api_app", factory=True, host=host, port=port)


@app.command("smoke-test")
def smoke_test() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        response = jarvis.runtime_service.route(
            TaskRequest(
                raw_input="/remember jarvis boot verification",
                metadata={"source": "smoke-test"},
            )
        )
        typer.echo(json.dumps({"status": "ok", "response": response.model_dump(mode="json")}, indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("action")
def run_action(action_name: str, payload: str = typer.Option("{}", "--payload")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.execute_action(action_name, _parse_json(payload))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("tool")
def run_tool(tool_name: str, payload: str = typer.Option("{}", "--payload")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.invoke_tool(tool_name, _parse_json(payload))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("models")
def list_models() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.list_models(), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("providers")
def providers() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(
            json.dumps(
                {
                    "providers": [entry.model_dump(mode="json") for entry in jarvis.runtime_service.model_health()],
                    "embedding_providers": [entry.model_dump(mode="json") for entry in jarvis.runtime_service.embedding_health()],
                },
                indent=2,
                default=str,
            )
        )
    finally:
        jarvis.stop()


@app.command("infer")
def infer(
    prompt: str,
    logical_model: str | None = typer.Option(None, "--model"),
    task_type: str = typer.Option("assistant", "--task-type"),
    temperature: float | None = typer.Option(None, "--temperature"),
    timeout_seconds: float | None = typer.Option(None, "--timeout"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        response = jarvis.runtime_service.infer_model(
            ModelRequest(
                prompt=prompt,
                messages=[{"role": "user", "content": prompt}],
                logical_model=logical_model,
                task_type=task_type,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        )
        typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("orchestrate")
def orchestrate(
    intent: str | None = typer.Option(None, "--intent"),
    query: str | None = typer.Option(None, "--query"),
    payload: str = typer.Option("{}", "--payload"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        request = OrchestrationRequest(intent=intent, query=query, payload=_parse_json(payload))
        response = jarvis.orchestrator.handle(request)
        typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("task")
def task(
    raw_input: str | None = typer.Option(None, "--input"),
    intent: str | None = typer.Option(None, "--intent"),
    payload: str = typer.Option("{}", "--payload"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        response = jarvis.runtime_service.route(
            TaskRequest(
                raw_input=raw_input,
                intent=intent,
                payload=_parse_json(payload),
            )
        )
        typer.echo(json.dumps(response.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("mode")
def mode(
    target: ExecutionMode | None = typer.Argument(None),
    reason: str | None = typer.Option(None, "--reason"),
    sticky: bool = typer.Option(True, "--sticky/--transient"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        if target is None:
            typer.echo(json.dumps(jarvis.mode_manager.snapshot().model_dump(mode="json"), indent=2, default=str))
            return
        snapshot = jarvis.runtime_service.switch_mode(target, reason=reason, sticky=sticky)
        typer.echo(json.dumps(snapshot.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("state")
def state() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.snapshot().model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("automation")
def save_automation(
    name: str,
    action_name: str,
    interval_seconds: int,
    payload: str = typer.Option("{}", "--payload"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        entry = jarvis.automation_service.save(
            AutomationDefinition(
                name=name,
                action_name=action_name,
                interval_seconds=interval_seconds,
                payload=_parse_json(payload),
            )
        )
        typer.echo(json.dumps(entry.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@app.command("describe")
def describe() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.describe(), indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("run")
def desktop_agent_run(
    goal: str,
    max_steps: int | None = typer.Option(None, "--max-steps"),
    detach: bool = typer.Option(False, "--detach", help="Create the mission and return immediately without waiting for completion."),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.desktop_agent_run(
            DesktopAgentMissionRequest(goal=goal, max_steps=max_steps, wait_for_completion=not detach)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("status")
def desktop_agent_status(mission: str | None = typer.Option(None, "--mission")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        if mission:
            typer.echo(json.dumps(jarvis.runtime_service.desktop_agent_get(mission).model_dump(mode="json"), indent=2, default=str))
        else:
            typer.echo(json.dumps(jarvis.runtime_service.desktop_agent_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("list")
def desktop_agent_list() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps([mission.model_dump(mode="json") for mission in jarvis.runtime_service.desktop_agent_list()], indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("pause")
def desktop_agent_pause(mission: str = typer.Option(..., "--mission")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.desktop_agent_pause(mission).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("resume")
def desktop_agent_resume(mission: str = typer.Option(..., "--mission")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.desktop_agent_resume(mission).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@desktop_agent_app.command("abort")
def desktop_agent_abort(mission: str = typer.Option(..., "--mission")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.desktop_agent_abort(mission).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@semantic_app.command("ingest")
def semantic_ingest(
    path: str | None = typer.Option(None, "--path"),
    content: str | None = typer.Option(None, "--content"),
    collection: str = typer.Option("default", "--collection"),
    source_type: str = typer.Option("text", "--source-type"),
    title: str | None = typer.Option(None, "--title"),
    metadata: str = typer.Option("{}", "--metadata"),
    persist_memory: bool = typer.Option(False, "--persist-memory"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        document = jarvis.runtime_service.semantic_ingest(
            DocumentIngestionRequest(
                collection_name=collection,
                source_type=source_type,
                path=path,
                content=content,
                title=title,
                metadata=_parse_json(metadata),
                persist_memory=persist_memory,
            )
        )
        typer.echo(json.dumps(document.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@semantic_app.command("search")
def semantic_search(
    query: str,
    collection: str | None = typer.Option(None, "--collection"),
    top_k: int | None = typer.Option(None, "--top-k"),
    min_score: float | None = typer.Option(None, "--min-score"),
    metadata_filters: str = typer.Option("{}", "--filters"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        result = jarvis.runtime_service.semantic_search(
            SemanticSearchQuery(
                query=query,
                collection_name=collection,
                top_k=top_k,
                min_score=min_score,
                metadata_filters=_parse_json(metadata_filters),
            )
        )
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@semantic_app.command("collections")
def semantic_collections() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.semantic_collections(), indent=2, default=str))
    finally:
        jarvis.stop()


@semantic_app.command("status")
def semantic_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.semantic_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@semantic_app.command("reindex")
def semantic_reindex(collection: str | None = typer.Option(None, "--collection")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.semantic_reindex(collection), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("status")
def ui_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.ui_automation_service.health(), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("active-window")
def ui_active_window() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ui_active_window().model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("focus")
def ui_focus(target: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ui_focus_window(FocusWindowRequest(target=target)).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("write")
def ui_write(
    text: str,
    mode: UIAutomationMode = typer.Option(UIAutomationMode.COPILOT, "--mode"),
    target_window: str | None = typer.Option(None, "--target-window"),
    ensure_window_contains: str | None = typer.Option(None, "--ensure-window"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(
            json.dumps(
                jarvis.runtime_service.ui_write_text(
                    WriteTextRequest(
                        text=text,
                        mode=mode,
                        focus_target=target_window,
                        ensure_window_contains=ensure_window_contains,
                    )
                ).model_dump(mode="json"),
                indent=2,
                default=str,
            )
        )
    finally:
        jarvis.stop()


@ui_app.command("move-mouse")
def ui_move_mouse(x: int, y: int, relative: bool = typer.Option(False, "--relative")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ui_move_mouse(MoveMouseRequest(x=x, y=y, relative=relative)).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("click")
def ui_click(
    button: str = typer.Option("left", "--button"),
    double: bool = typer.Option(False, "--double"),
    x: int | None = typer.Option(None, "--x"),
    y: int | None = typer.Option(None, "--y"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ui_click(ClickRequest(button=button, double=double, x=x, y=y)).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("hotkey")
def ui_hotkey(keys: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        parsed = tuple(item.strip() for item in keys.split("+") if item.strip())
        typer.echo(json.dumps(jarvis.runtime_service.ui_hotkey(ShortcutRequest(keys=parsed)).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ui_app.command("cancel")
def ui_cancel(correlation_id: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ui_cancel(correlation_id).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("status")
def voice_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("listen")
def voice_listen(
    duration_seconds: float | None = typer.Option(None, "--seconds"),
    language: str | None = typer.Option(None, "--language"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.voice_start_session(
            VoiceSessionRequest(duration_seconds=duration_seconds, language=language)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("stop")
def voice_stop() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_stop_session().model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("transcribe")
def voice_transcribe(file_path: str = typer.Option(..., "--file")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_transcribe_file(file_path).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("speak")
def voice_speak(text: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_speak(text).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("dictate")
def voice_dictate(
    mode: str = typer.Option("copilot", "--mode"),
    duration_seconds: float | None = typer.Option(None, "--seconds"),
    target_window: str | None = typer.Option(None, "--target-window"),
    language: str | None = typer.Option(None, "--language"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.voice_dictate(
            VoiceSessionRequest(
                mode="dictation",
                duration_seconds=duration_seconds,
                target_window=target_window,
                language=language,
                ui_mode=mode,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("clap-status")
def voice_clap_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_clap_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@voice_app.command("cancel")
def voice_cancel(correlation_id: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.voice_cancel(correlation_id).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("status")
def system_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.system_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("search")
def system_search(query: str, kind: str | None = typer.Option(None, "--kind"), scope: str = typer.Option("all", "--scope"), roots: str | None = typer.Option(None, "--roots"), extensions: str | None = typer.Option(None, "--extensions"), max_results: int = typer.Option(10, "--max-results")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.system_search(
            SystemSearchRequest(
                resource=ResourceQuery(
                    query=query,
                    target_kind=kind,
                    search_scope=scope,
                    preferred_roots=[item.strip() for item in roots.split(",")] if roots else [],
                    extensions=[item.strip() for item in extensions.split(",")] if extensions else [],
                    max_results=max_results,
                )
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("resolve")
def system_resolve(query: str, kind: str | None = typer.Option(None, "--kind"), scope: str = typer.Option("all", "--scope"), roots: str | None = typer.Option(None, "--roots"), extensions: str | None = typer.Option(None, "--extensions")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.system_resolve(
            SystemResolveRequest(
                query=query,
                target_kind=kind,
                search_scope=scope,
                preferred_roots=[item.strip() for item in roots.split(",")] if roots else [],
                extensions=[item.strip() for item in extensions.split(",")] if extensions else [],
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("open")
def system_open(query: str | None = typer.Option(None, "--query"), path: str | None = typer.Option(None, "--path"), uri: str | None = typer.Option(None, "--uri"), dry_run: bool = typer.Option(False, "--dry-run"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.system_open(
            SystemOpenRequest(query=query, path=path, uri=uri, dry_run=dry_run, metadata={"approved": approved})
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("open-app")
def system_open_app(application: str, dry_run: bool = typer.Option(False, "--dry-run"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.system_open_application(application, dry_run=dry_run, metadata={"approved": approved})
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@system_app.command("reveal")
def system_reveal(path: str, dry_run: bool = typer.Option(False, "--dry-run"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.system_reveal(path, dry_run=dry_run, metadata={"approved": approved})
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("status")
def unity_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.unity_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("resolve-project")
def unity_resolve_project(query: str, roots: str | None = typer.Option(None, "--roots")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_resolve_project(
            UnityProjectResolveRequest(query=UnityProjectQuery(query=query, preferred_roots=[item.strip() for item in roots.split(",")] if roots else []))
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("create-project")
def unity_create_project(name: str, root: str = typer.Option(..., "--root"), template: str = typer.Option("3d", "--template"), version: str | None = typer.Option(None, "--version"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_create_project(
            UnityProjectCreateRequest(name=name, target_root=root, template=template, unity_version=version, metadata={"approved": approved})
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("list-scenes")
def unity_list_scenes(project: str = typer.Option(..., "--project")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_list_scenes(project)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("search-assets")
def unity_search_assets(query: str, project: str = typer.Option(..., "--project"), kind: str | None = typer.Option(None, "--kind"), limit: int = typer.Option(20, "--limit")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_search_assets(UnityAssetSearchRequest(project=project, query=query, asset_kind=kind, limit=limit))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("generate-script")
def unity_generate_script(class_name: str, project: str = typer.Option(..., "--project"), folder: str | None = typer.Option(None, "--folder"), namespace: str | None = typer.Option(None, "--namespace"), script_type: str = typer.Option("mono_behaviour", "--type"), overwrite: bool = typer.Option(False, "--overwrite")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_generate_script(
            UnityScriptGenerationRequest(project=project, folder=folder, class_name=class_name, namespace=namespace, script_type=script_type, overwrite=overwrite)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("write-script")
def unity_write_script(project: str = typer.Option(..., "--project"), path: str | None = typer.Option(None, "--path"), folder: str | None = typer.Option(None, "--folder"), class_name: str | None = typer.Option(None, "--class-name"), content: str | None = typer.Option(None, "--content"), content_file: str | None = typer.Option(None, "--content-file"), overwrite: bool = typer.Option(False, "--overwrite"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        final_content = content
        if content_file:
            final_content = Path(content_file).read_text(encoding="utf-8")
        receipt = jarvis.runtime_service.unity_write_script(
            UnityScriptWriteRequest(project=project, asset_path=path, folder=folder, class_name=class_name, content=final_content or "", overwrite=overwrite, metadata={"approved": approved})
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("open-project")
def unity_open_project(project: str, approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_open_project(project, metadata={"approved": approved})
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("launch-project")
def unity_launch_project(
    project: str = typer.Option(..., "--project"),
    strategy: str | None = typer.Option(None, "--strategy"),
    installation_id: str | None = typer.Option(None, "--installation-id"),
    approved: bool = typer.Option(False, "--approved"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_launch_project(
            UnityLaunchRequestModel(project=project, installation_id=installation_id, strategy=strategy, metadata={"approved": approved})
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("bridge-status")
def unity_bridge_status(project: str | None = typer.Option(None, "--project")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_bridge_health(project)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("bridge-connect")
def unity_bridge_connect(
    project: str = typer.Option(..., "--project"),
    endpoint: str | None = typer.Option(None, "--endpoint"),
    installation_id: str | None = typer.Option(None, "--installation-id"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_connect_bridge(
            UnityBridgeConnectRequest(project=project, endpoint=endpoint, installation_id=installation_id)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("bridge-disconnect")
def unity_bridge_disconnect(project: str = typer.Option(..., "--project")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_disconnect_bridge(UnityBridgeDisconnectRequest(project=project))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("bridge")
def unity_bridge(project: str = typer.Option(..., "--project"), command: str = typer.Option(..., "--command"), payload: str = typer.Option("{}", "--payload"), approved: bool = typer.Option(False, "--approved")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_bridge_call(
            UnityBridgeRequest(project=project, command=command, payload=_parse_json(payload), metadata={"approved": approved})
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@unity_app.command("editor-command")
def unity_editor_command(
    project: str = typer.Option(..., "--project"),
    operation: str = typer.Option(..., "--operation"),
    payload: str = typer.Option("{}", "--payload"),
    scene: str | None = typer.Option(None, "--scene"),
    asset_path: str | None = typer.Option(None, "--asset-path"),
    approved: bool = typer.Option(False, "--approved"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.unity_editor_operation(
            UnityEditorOperationRequest(
                project=project,
                operation_kind=UnityEditorOperationKind(operation),
                scene=scene,
                asset_path=asset_path,
                parameters=_parse_json(payload),
                metadata={"approved": approved},
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@research_app.command("run")
def research_run(
    topic: str,
    collection: str | None = typer.Option(None, "--collection"),
    paths: str | None = typer.Option(None, "--paths"),
    images: str | None = typer.Option(None, "--images"),
    simulated: str = typer.Option("[]", "--simulated"),
    via_autonomy: bool = typer.Option(False, "--via-autonomy"),
    autonomy_level: str | None = typer.Option(None, "--autonomy-level"),
    max_steps: int = typer.Option(12, "--max-steps"),
    max_sources: int = typer.Option(6, "--max-sources"),
    max_findings: int = typer.Option(20, "--max-findings"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        parsed_simulated = [SimulatedResearchSource.model_validate(item) for item in json.loads(simulated)]
        task = jarvis.runtime_service.research_run(
            ResearchRunRequest(
                query=topic,
                collection_name=collection,
                paths=[item.strip() for item in paths.split(",")] if paths else [],
                image_paths=[item.strip() for item in images.split(",")] if images else [],
                simulated_sources=parsed_simulated,
                run_via_autonomy=via_autonomy,
                autonomy_level=autonomy_level,
                budget=ResearchBudget(max_steps=max_steps, max_sources=max_sources, max_findings=max_findings),
            )
        )
        typer.echo(json.dumps(task.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@research_app.command("status")
def research_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.research_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@research_app.command("report")
def research_report(task_id: str | None = typer.Option(None, "--task-id")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.research_report(task_id).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@science_app.command("solve")
def science_solve(
    query: str,
    operation: str | None = typer.Option(None, "--operation"),
    parameters: str = typer.Option("{}", "--parameters"),
    plot: bool = typer.Option(False, "--plot"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.science_solve(
            ScienceSolveRequest(
                query=query,
                operation=operation,
                parameters=_parse_json(parameters),
                generate_plot=plot,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@science_app.command("simulate")
def science_simulate(
    query: str | None = None,
    simulation_type: str | None = typer.Option(None, "--type"),
    parameters: str = typer.Option("{}", "--parameters"),
    duration: float | None = typer.Option(None, "--duration"),
    time_step: float | None = typer.Option(None, "--time-step"),
    max_points: int = typer.Option(2000, "--max-points"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.science_simulate(
            ScienceSimulationRequest(
                query=query,
                simulation_type=simulation_type,
                parameters=_parse_json(parameters),
                duration=duration,
                time_step=time_step,
                max_points=max_points,
                generate_plot=plot,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@security_app.command("analyze")
def security_analyze(
    query: str | None = typer.Option(None, "--query"),
    code: str | None = typer.Option(None, "--code"),
    path: str | None = typer.Option(None, "--path"),
    audit_kind: str | None = typer.Option(None, "--audit-kind"),
    include_workspace: bool = typer.Option(False, "--include-workspace"),
    max_findings: int = typer.Option(50, "--max-findings"),
    host: str = typer.Option("127.0.0.1", "--host"),
    ports: str | None = typer.Option(None, "--ports"),
    timeout_ms: int = typer.Option(250, "--timeout-ms"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        parsed_ports = [int(item.strip()) for item in ports.split(",")] if ports else []
        receipt = jarvis.runtime_service.security_analyze(
            SecurityAnalyzeRequest(
                query=query,
                code=code,
                path=path,
                audit_kind=audit_kind,
                include_workspace=include_workspace,
                max_findings=max_findings,
                host=host,
                ports=parsed_ports,
                timeout_ms=timeout_ms,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@security_app.command("check-password")
def security_check_password(password: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.security_check_password(SecurityPasswordCheckRequest(password=password))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@index_app.command("status")
def index_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.indexing_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@index_app.command("run")
def index_run(
    sources: str | None = typer.Option(None, "--sources"),
    trigger: str = typer.Option(IndexingTrigger.MANUAL.value, "--trigger"),
    requested_by: str | None = typer.Option(None, "--requested-by"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.indexing_run(
            IndexRunRequest(
                source_ids=tuple(item.strip() for item in sources.split(",") if item.strip()) if sources else (),
                trigger=IndexingTrigger(trigger),
                requested_by=requested_by,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@index_app.command("reindex")
def index_reindex(
    sources: str | None = typer.Option(None, "--sources"),
    trigger: str = typer.Option(IndexingTrigger.MANUAL.value, "--trigger"),
    requested_by: str | None = typer.Option(None, "--requested-by"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.indexing_reindex(
            IndexRunRequest(
                source_ids=tuple(item.strip() for item in sources.split(",") if item.strip()) if sources else (),
                trigger=IndexingTrigger(trigger),
                force_reindex=True,
                requested_by=requested_by,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@index_app.command("add-source")
def index_add_source(
    id: str = typer.Option(..., "--id"),
    kind: str = typer.Option(..., "--kind"),
    name: str = typer.Option(..., "--name"),
    root: str | None = typer.Option(None, "--root"),
    collection: str | None = typer.Option(None, "--collection"),
    enabled: bool = typer.Option(True, "--enabled/--disabled"),
    priority: int = typer.Option(100, "--priority"),
    patterns: str | None = typer.Option(None, "--patterns"),
    exclude: str | None = typer.Option(None, "--exclude"),
    extensions: str | None = typer.Option(None, "--extensions"),
    max_file_size: int | None = typer.Option(None, "--max-file-size"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.indexing_add_source(
            IndexSourceCreateRequest(
                source_id=id,
                source_kind=IndexSourceKind(kind),
                display_name=name,
                root_path=root,
                collection_name=collection,
                enabled=enabled,
                priority=priority,
                file_patterns=tuple(item.strip() for item in patterns.split(",") if item.strip()) if patterns else ("*",),
                exclude_patterns=tuple(item.strip() for item in exclude.split(",") if item.strip()) if exclude else (),
                allowed_extensions=tuple(item.strip() for item in extensions.split(",") if item.strip()) if extensions else (),
                max_file_size_bytes=max_file_size,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@writing_app.command("continue")
def writing_continue(
    prompt: str,
    instruction: str | None = typer.Option(None, "--instruction"),
    target_window: str | None = typer.Option(None, "--target-window"),
    desired_words: int = typer.Option(120, "--words"),
    collection: str | None = typer.Option(None, "--collection"),
    write_directly: bool = typer.Option(True, "--write/--no-write"),
) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.writing_continue(
            WritingContinuationRequest(
                prompt=prompt,
                instruction=instruction,
                mode=WritingMode.COPILOT,
                target_window=target_window,
                desired_words=desired_words,
                write_directly=write_directly,
                collection_name=collection,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@writing_app.command("analyze")
def writing_analyze(prompt: str, target_window: str | None = typer.Option(None, "--target-window"), collection: str | None = typer.Option(None, "--collection")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        result = jarvis.runtime_service.writing_analyze(
            WritingContinuationRequest(prompt=prompt, target_window=target_window, collection_name=collection, write_directly=False)
        )
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@writing_app.command("autonomous-start")
def writing_autonomous_start(prompt: str, target_window: str | None = typer.Option(None, "--target-window"), desired_words: int = typer.Option(180, "--words"), collection: str | None = typer.Option(None, "--collection")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.writing_autonomous_start(
            WritingContinuationRequest(
                prompt=prompt,
                mode=WritingMode.AUTONOMOUS,
                target_window=target_window,
                desired_words=desired_words,
                collection_name=collection,
            )
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@writing_app.command("autonomous-stop")
def writing_autonomous_stop(task_id: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.writing_autonomous_stop(task_id)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("status")
def ops_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ops_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("health")
def ops_health() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        probes = jarvis.runtime_service.ops_health()
        typer.echo(json.dumps([probe.model_dump(mode="json") for probe in probes], indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("snapshot")
def ops_snapshot() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ops_snapshot().model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("diagnostics")
def ops_diagnostics(service: str | None = typer.Option(None, "--service")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        reports = jarvis.runtime_service.ops_diagnostics(service)
        typer.echo(json.dumps([report.model_dump(mode="json") for report in reports], indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("recover")
def ops_recover(service: str = typer.Option(..., "--service"), dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ops_recover_service(service, dry_run=dry_run).model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("reset-breaker")
def ops_reset_breaker(service: str = typer.Option(..., "--service"), dependency: str | None = typer.Option(None, "--dependency")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ops_reset_breaker(service, dependency), indent=2, default=str))
    finally:
        jarvis.stop()


@ops_app.command("retention-sweep")
def ops_retention_sweep() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.ops_retention_sweep().model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("status")
def autonomy_status() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.autonomy_status(), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("plan")
def autonomy_plan(goal: str, payload: str = typer.Option("{}", "--payload"), level: str | None = typer.Option(None, "--level")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        plan = jarvis.runtime_service.autonomy_plan(MissionPlanRequest(goal=goal, payload=_parse_json(payload), autonomy_level=level))
        typer.echo(json.dumps(plan.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("start")
def autonomy_start(goal: str, payload: str = typer.Option("{}", "--payload"), level: str | None = typer.Option(None, "--level")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_start(MissionRequest(goal=goal, payload=_parse_json(payload), autonomy_level=level))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("step")
def autonomy_step(mission: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_step(mission)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("run")
def autonomy_run(mission: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_run(mission)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("stop")
def autonomy_stop(mission: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_stop(mission)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("approve")
def autonomy_approve(mission: str, step: str | None = typer.Option(None, "--step"), reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_approve(
            MissionApprovalRequest(mission_id=mission, step_id=step, decision="approve", reason=reason, actor=actor)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("reject")
def autonomy_reject(mission: str, step: str | None = typer.Option(None, "--step"), reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_reject(
            MissionApprovalRequest(mission_id=mission, step_id=step, decision="reject", reason=reason, actor=actor)
        )
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("pause")
def autonomy_pause(mission: str, reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_pause(MissionControlActionRequest(mission_id=mission, reason=reason, actor=actor))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("resume")
def autonomy_resume(mission: str, reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_resume(MissionControlActionRequest(mission_id=mission, reason=reason, actor=actor))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("retry-step")
def autonomy_retry_step(mission: str, step: str | None = typer.Option(None, "--step"), reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_retry_step(MissionControlActionRequest(mission_id=mission, step_id=step, reason=reason, actor=actor))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("skip-step")
def autonomy_skip_step(mission: str, step: str | None = typer.Option(None, "--step"), reason: str | None = typer.Option(None, "--reason"), actor: str | None = typer.Option("cli", "--actor")) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_skip_step(MissionControlActionRequest(mission_id=mission, step_id=step, reason=reason, actor=actor))
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("missions")
def autonomy_missions() -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        typer.echo(json.dumps(jarvis.runtime_service.autonomy_missions(), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("inspect")
def autonomy_inspect(mission: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        receipt = jarvis.runtime_service.autonomy_inspect(mission)
        typer.echo(json.dumps(receipt.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()


@autonomy_app.command("control")
def autonomy_control(mission: str) -> None:
    jarvis = build_application()
    jarvis.start()
    try:
        view = jarvis.runtime_service.autonomy_control_view(mission)
        typer.echo(json.dumps(view.model_dump(mode="json"), indent=2, default=str))
    finally:
        jarvis.stop()
