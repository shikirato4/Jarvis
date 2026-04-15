from __future__ import annotations

from typing import Any

from jarvis.actions.router import ActionRouter
from jarvis.autonomy.base import MissionApprovalRequest, MissionControlActionRequest, MissionPlanRequest, MissionRequest
from jarvis.autonomy.service import AutonomyService
from jarvis.automation.service import AutomationDefinition, AutomationService
from jarvis.config import Settings
from jarvis.core.errors import ServiceUnavailableError
from jarvis.core.models import HealthStatus, RuntimeSnapshot, ServiceStatus
from jarvis.core.modes import ExecutionMode, ModeManager
from jarvis.core.services import RuntimeServiceContract
from jarvis.core.state import RuntimeStateManager
from jarvis.memory.service import MemoryService
from jarvis.memory_semantic.base import SemanticSearchQuery
from jarvis.memory_semantic.documents import DocumentIngestionRequest
from jarvis.memory_semantic.embeddings import EmbeddingProviderRegistry, EmbeddingRouter, EmbeddingService
from jarvis.memory_semantic.service import SemanticMemoryService
from jarvis.indexing_runtime.models import IndexRunRequest, IndexSourceCreateRequest
from jarvis.models_runtime.base import ModelRequest
from jarvis.models_runtime.catalog import ModelCatalog
from jarvis.models_runtime.registry import ProviderRegistry
from jarvis.models_runtime.router import ModelRouter
from jarvis.models_runtime.service import ModelService
from jarvis.research_runtime.models import ResearchRunRequest
from jarvis.research_runtime.service import ResearchRuntimeService
from jarvis.routing.models import TaskRequest, TaskResponse
from jarvis.routing.task_router import TaskRouter
from jarvis.science_runtime import ScienceSimulationRequest, ScienceSolveRequest
from jarvis.science_runtime.service import ScienceRuntimeService
from jarvis.security_runtime import SecurityAnalyzeRequest, SecurityPasswordCheckRequest
from jarvis.security_runtime.service import SecurityRuntimeService
from jarvis.self_improvement_runtime.models import SelfImprovementRequest
from jarvis.self_improvement_runtime.service import SelfImprovementRuntimeService
from jarvis.tools.registry import ToolRegistry
from jarvis.unity_runtime.base import (
    UnityAssetSearchRequest,
    UnityBridgeConnectRequest,
    UnityBridgeDisconnectRequest,
    UnityBridgeRequest,
    UnityEditorOperationRequest,
    UnityLaunchRequestModel,
    UnityProjectCreateRequest,
    UnityProjectResolveRequest,
    UnityScriptGenerationRequest,
    UnityScriptWriteRequest,
)
from jarvis.unity_runtime.service import UnityRuntimeService
from jarvis.ui_automation.base import CancellationRequest, ClickRequest, ClickVisualTargetRequest, CloseWindowRequest, FocusWindowRequest, MoveMouseRequest, ShortcutRequest, WriteTextRequest
from jarvis.ui_automation.service import UIAutomationService
from jarvis.system_runtime.base import SystemOpenRequest, SystemResolveRequest, SystemSearchRequest
from jarvis.desktop_agent_runtime import DesktopAgentMissionRequest
from jarvis.system_runtime.service import SystemRuntimeService
from jarvis.vision_runtime.base import ElementLocationRequest, OCRRequest, ScreenCaptureRequest, TextLocationRequest, UIAwarenessRequest, VisionAnalysisRequest
from jarvis.vision_runtime.service import VisionRuntimeService
from jarvis.voice_runtime.base import VoiceSessionRequest
from jarvis.voice_runtime.service import VoiceRuntimeService
from jarvis.writing_runtime.models import WritingContinuationRequest
from jarvis.writing_runtime.service import WritingRuntimeService


class JarvisRuntimeService(RuntimeServiceContract):
    service_name = "runtime"

    def __init__(
        self,
        *,
        settings: Settings,
        mode_manager: ModeManager,
        state_manager: RuntimeStateManager,
        task_router: TaskRouter,
        action_router: ActionRouter,
        tool_registry: ToolRegistry,
        provider_registry: ProviderRegistry,
        model_catalog: ModelCatalog,
        model_router: ModelRouter,
        model_service: ModelService,
        embedding_provider_registry: EmbeddingProviderRegistry,
        embedding_router: EmbeddingRouter,
        embedding_service: EmbeddingService,
        ui_automation_service: UIAutomationService,
        vision_runtime_service: VisionRuntimeService,
        voice_runtime_service: VoiceRuntimeService,
        system_runtime_service: SystemRuntimeService,
        unity_runtime_service: UnityRuntimeService,
        autonomy_service: AutonomyService,
        research_runtime_service: ResearchRuntimeService,
        science_runtime_service: ScienceRuntimeService,
        security_runtime_service: SecurityRuntimeService,
        self_improvement_runtime_service: SelfImprovementRuntimeService,
        writing_runtime_service: WritingRuntimeService,
        memory_service: MemoryService,
        semantic_memory_service: SemanticMemoryService,
        indexing_runtime_service,
        automation_service: AutomationService,
        ops_runtime_service=None,
        desktop_agent_runtime_service=None,
    ) -> None:
        self._settings = settings
        self._mode_manager = mode_manager
        self._state_manager = state_manager
        self._task_router = task_router
        self._action_router = action_router
        self._tool_registry = tool_registry
        self._provider_registry = provider_registry
        self._model_catalog = model_catalog
        self._model_router = model_router
        self._model_service = model_service
        self._embedding_provider_registry = embedding_provider_registry
        self._embedding_router = embedding_router
        self._embedding_service = embedding_service
        self._ui_automation_service = ui_automation_service
        self._vision_runtime_service = vision_runtime_service
        self._voice_runtime_service = voice_runtime_service
        self._system_runtime_service = system_runtime_service
        self._unity_runtime_service = unity_runtime_service
        self._autonomy_service = autonomy_service
        self._research_runtime_service = research_runtime_service
        self._science_runtime_service = science_runtime_service
        self._security_runtime_service = security_runtime_service
        self._self_improvement_runtime_service = self_improvement_runtime_service
        self._writing_runtime_service = writing_runtime_service
        self._memory_service = memory_service
        self._semantic_memory_service = semantic_memory_service
        self._indexing_runtime_service = indexing_runtime_service
        self._automation_service = automation_service
        self._ops_runtime_service = ops_runtime_service
        self._desktop_agent_runtime_service = desktop_agent_runtime_service
        self._started = False

    def start(self) -> None:
        self._started = True
        self._state_manager.update_service(self.service_name, HealthStatus.READY)

    def stop(self) -> None:
        self._started = False
        self._state_manager.update_service(self.service_name, HealthStatus.STOPPED)

    def health(self) -> ServiceStatus:
        return ServiceStatus(
            name=self.service_name,
            status=HealthStatus.READY if self._started else HealthStatus.STOPPED,
            details={"mode": self._mode_manager.current_mode().value},
        )

    def switch_mode(self, mode: ExecutionMode | str, *, reason: str | None = None, sticky: bool = True) -> RuntimeSnapshot:
        self._ensure_started()
        self._mode_manager.set_mode(mode, reason=reason, sticky=sticky)
        return self.snapshot()

    def route(self, request: TaskRequest | dict[str, Any]) -> TaskResponse:
        self._ensure_started()
        return self._task_router.route(TaskRequest.model_validate(request))

    def execute_action(
        self,
        action_name: str,
        payload: dict[str, Any],
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self._ensure_started()
        return self._action_router.execute(action_name, payload, dry_run=dry_run, metadata=metadata)

    def invoke_tool(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        dry_run: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        self._ensure_started()
        return self._tool_registry.invoke(tool_name, payload, dry_run=dry_run, metadata=metadata)

    def save_automation(self, definition: AutomationDefinition | dict[str, Any]):
        self._ensure_started()
        return self._automation_service.save(definition)

    def infer_model(self, request: ModelRequest | dict[str, Any]):
        self._ensure_started()
        return self._model_service.infer(ModelRequest.model_validate(request))

    def model_health(self):
        self._ensure_started()
        return self._model_service.health()

    def list_models(self) -> list[dict[str, Any]]:
        self._ensure_started()
        return self._model_service.list_models()

    def list_embedding_models(self) -> list[dict[str, Any]]:
        self._ensure_started()
        return self._embedding_service.list_models()

    def embedding_health(self):
        self._ensure_started()
        return self._embedding_service.health()

    def semantic_ingest(self, request: DocumentIngestionRequest | dict[str, Any]):
        self._ensure_started()
        return self._semantic_memory_service.ingest_document(DocumentIngestionRequest.model_validate(request))

    def semantic_search(self, query: SemanticSearchQuery | dict[str, Any]):
        self._ensure_started()
        return self._semantic_memory_service.search(SemanticSearchQuery.model_validate(query))

    def semantic_collections(self) -> list[dict[str, Any]]:
        self._ensure_started()
        return self._semantic_memory_service.list_collections()

    def semantic_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._semantic_memory_service.status()

    def semantic_reindex(self, collection_name: str | None = None) -> dict[str, Any]:
        self._ensure_started()
        return self._semantic_memory_service.reindex(collection_name)

    def indexing_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._indexing_runtime_service.status()

    def indexing_run(self, request: IndexRunRequest | dict[str, Any]):
        self._ensure_started()
        return self._indexing_runtime_service.run(IndexRunRequest.model_validate(request))

    def indexing_reindex(self, request: IndexRunRequest | dict[str, Any]):
        self._ensure_started()
        return self._indexing_runtime_service.reindex(IndexRunRequest.model_validate(request))

    def indexing_add_source(self, request: IndexSourceCreateRequest | dict[str, Any]):
        self._ensure_started()
        return self._indexing_runtime_service.add_source(IndexSourceCreateRequest.model_validate(request))

    def ui_active_window(self):
        self._ensure_started()
        return self._ui_automation_service.active_window(correlation_id="runtime-ui-active-window")

    def ui_focus_window(self, request: FocusWindowRequest | dict[str, Any]):
        self._ensure_started()
        return self._ui_automation_service.focus_window(FocusWindowRequest.model_validate(request), correlation_id="runtime-ui-focus")

    def ui_close_window(self, request: CloseWindowRequest | dict[str, Any]):
        self._ensure_started()
        return self._ui_automation_service.close_window(CloseWindowRequest.model_validate(request), correlation_id="runtime-ui-close")

    def ui_write_text(self, request: WriteTextRequest | dict[str, Any], *, correlation_id: str = "runtime-ui-write"):
        self._ensure_started()
        return self._ui_automation_service.write_text(WriteTextRequest.model_validate(request), correlation_id=correlation_id)

    def ui_move_mouse(self, request: MoveMouseRequest | dict[str, Any]):
        self._ensure_started()
        return self._ui_automation_service.move_mouse(MoveMouseRequest.model_validate(request), correlation_id="runtime-ui-mouse")

    def ui_click(self, request: ClickRequest | dict[str, Any]):
        self._ensure_started()
        return self._ui_automation_service.click(ClickRequest.model_validate(request), correlation_id="runtime-ui-click")

    def ui_click_target(self, request: ClickVisualTargetRequest | dict[str, Any], *, correlation_id: str = "runtime-ui-click-target"):
        self._ensure_started()
        return self._ui_automation_service.click_visual_target(
            ClickVisualTargetRequest.model_validate(request),
            correlation_id=correlation_id,
            vision_runtime=self._vision_runtime_service,
        )

    def ui_hotkey(self, request: ShortcutRequest | dict[str, Any]):
        self._ensure_started()
        return self._ui_automation_service.hotkey(ShortcutRequest.model_validate(request), correlation_id="runtime-ui-hotkey")

    def ui_cancel(self, correlation_id: str):
        self._ensure_started()
        return self._ui_automation_service.cancel(CancellationRequest(correlation_id=correlation_id))

    def voice_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._voice_runtime_service.status()

    def system_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._system_runtime_service.status()

    def system_search(self, request: SystemSearchRequest | dict[str, Any]):
        self._ensure_started()
        return self._system_runtime_service.search(SystemSearchRequest.model_validate(request))

    def system_resolve(self, request: SystemResolveRequest | dict[str, Any]):
        self._ensure_started()
        return self._system_runtime_service.resolve(SystemResolveRequest.model_validate(request))

    def system_open(self, request: SystemOpenRequest | dict[str, Any]):
        self._ensure_started()
        return self._system_runtime_service.open(SystemOpenRequest.model_validate(request))

    def system_open_path(self, path: str, *, reveal_in_folder: bool = False, dry_run: bool = False, metadata: dict[str, Any] | None = None):
        self._ensure_started()
        return self._system_runtime_service.open_path(path, reveal_in_folder=reveal_in_folder, dry_run=dry_run, metadata=metadata)

    def system_open_application(self, application: str, *, dry_run: bool = False, metadata: dict[str, Any] | None = None):
        self._ensure_started()
        return self._system_runtime_service.open_application(application, dry_run=dry_run, metadata=metadata)

    def system_reveal(self, path: str, *, dry_run: bool = False, metadata: dict[str, Any] | None = None):
        self._ensure_started()
        return self._system_runtime_service.reveal(path, dry_run=dry_run, metadata=metadata)

    def unity_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._unity_runtime_service.status()

    def unity_resolve_project(self, request: UnityProjectResolveRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.resolve_project(UnityProjectResolveRequest.model_validate(request))

    def unity_create_project(self, request: UnityProjectCreateRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.create_project(UnityProjectCreateRequest.model_validate(request))

    def unity_search_assets(self, request: UnityAssetSearchRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.search_assets(UnityAssetSearchRequest.model_validate(request))

    def unity_list_scenes(self, project: str):
        self._ensure_started()
        return self._unity_runtime_service.list_scenes(project)

    def unity_generate_script(self, request: UnityScriptGenerationRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.generate_script(UnityScriptGenerationRequest.model_validate(request))

    def unity_write_script(self, request: UnityScriptWriteRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.write_script(UnityScriptWriteRequest.model_validate(request))

    def unity_open_project(self, project: str, *, metadata: dict[str, Any] | None = None):
        self._ensure_started()
        return self._unity_runtime_service.open_project(project, metadata=metadata)

    def unity_launch_project(self, request: UnityLaunchRequestModel | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.launch_project(UnityLaunchRequestModel.model_validate(request))

    def unity_open_editor(self, request: UnityLaunchRequestModel | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.open_editor(UnityLaunchRequestModel.model_validate(request))

    def unity_bridge_health(self, project: str | None = None):
        self._ensure_started()
        return self._unity_runtime_service.bridge_health(project)

    def unity_connect_bridge(self, request: UnityBridgeConnectRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.connect_bridge(UnityBridgeConnectRequest.model_validate(request))

    def unity_disconnect_bridge(self, request: UnityBridgeDisconnectRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.disconnect_bridge(UnityBridgeDisconnectRequest.model_validate(request))

    def unity_editor_operation(self, request: UnityEditorOperationRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.editor_operation(UnityEditorOperationRequest.model_validate(request))

    def unity_editor_command(self, request: UnityEditorOperationRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.editor_operation(UnityEditorOperationRequest.model_validate(request))

    def unity_bridge_call(self, request: UnityBridgeRequest | dict[str, Any]):
        self._ensure_started()
        return self._unity_runtime_service.bridge_call(UnityBridgeRequest.model_validate(request))

    def autonomy_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._autonomy_service.status()

    def autonomy_plan(self, request: MissionPlanRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.plan_mission(MissionPlanRequest.model_validate(request))

    def autonomy_start(self, request: MissionRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.start_mission(MissionRequest.model_validate(request))

    def autonomy_step(self, mission_id: str):
        self._ensure_started()
        return self._autonomy_service.step_mission(mission_id)

    def autonomy_stop(self, mission_id: str):
        self._ensure_started()
        return self._autonomy_service.stop_mission(mission_id)

    def autonomy_approve(self, request: MissionApprovalRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.approve_step(MissionApprovalRequest.model_validate(request))

    def autonomy_reject(self, request: MissionApprovalRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.reject_step(MissionApprovalRequest.model_validate(request))

    def autonomy_pause(self, request: MissionControlActionRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.pause_mission(MissionControlActionRequest.model_validate(request))

    def autonomy_resume(self, request: MissionControlActionRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.resume_mission(MissionControlActionRequest.model_validate(request))

    def autonomy_retry_step(self, request: MissionControlActionRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.retry_step(MissionControlActionRequest.model_validate(request))

    def autonomy_skip_step(self, request: MissionControlActionRequest | dict[str, Any]):
        self._ensure_started()
        return self._autonomy_service.skip_step(MissionControlActionRequest.model_validate(request))

    def autonomy_control_view(self, mission_id: str):
        self._ensure_started()
        return self._autonomy_service.mission_control_view(mission_id)

    def autonomy_missions(self) -> list[dict[str, object]]:
        self._ensure_started()
        return self._autonomy_service.list_missions()

    def autonomy_inspect(self, mission_id: str):
        self._ensure_started()
        return self._autonomy_service.inspect_mission(mission_id)

    def autonomy_run(self, mission_id: str):
        self._ensure_started()
        return self._autonomy_service.run_mission(mission_id)

    def research_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._research_runtime_service.status()

    def research_run(self, request: ResearchRunRequest | dict[str, Any]):
        self._ensure_started()
        return self._research_runtime_service.run(ResearchRunRequest.model_validate(request))

    def research_task(self, task_id: str):
        self._ensure_started()
        return self._research_runtime_service.get_task(task_id)

    def research_report(self, task_id: str | None = None):
        self._ensure_started()
        return self._research_runtime_service.report(task_id)

    def science_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._science_runtime_service.status()

    def science_solve(self, request: ScienceSolveRequest | dict[str, Any]):
        self._ensure_started()
        return self._science_runtime_service.solve(ScienceSolveRequest.model_validate(request))

    def science_simulate(self, request: ScienceSimulationRequest | dict[str, Any]):
        self._ensure_started()
        return self._science_runtime_service.simulate(ScienceSimulationRequest.model_validate(request))

    def security_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._security_runtime_service.status()

    def security_analyze(self, request: SecurityAnalyzeRequest | dict[str, Any]):
        self._ensure_started()
        return self._security_runtime_service.analyze(SecurityAnalyzeRequest.model_validate(request))

    def security_check_password(self, request: SecurityPasswordCheckRequest | dict[str, Any]):
        self._ensure_started()
        return self._security_runtime_service.check_password(SecurityPasswordCheckRequest.model_validate(request))

    def self_improvement_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._self_improvement_runtime_service.status()

    def self_improvement_analyze(self, request: SelfImprovementRequest | dict[str, Any]):
        self._ensure_started()
        return self._self_improvement_runtime_service.analyze_code(SelfImprovementRequest.model_validate(request))

    def self_improvement_run(self, request: SelfImprovementRequest | dict[str, Any]):
        self._ensure_started()
        return self._self_improvement_runtime_service.run(SelfImprovementRequest.model_validate(request))

    def self_improvement_rollback(self, session_id: str):
        self._ensure_started()
        return self._self_improvement_runtime_service.rollback(session_id)

    def writing_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._writing_runtime_service.status()

    def writing_analyze(self, request: WritingContinuationRequest | dict[str, Any]):
        self._ensure_started()
        return self._writing_runtime_service.analyze(WritingContinuationRequest.model_validate(request))

    def writing_continue(self, request: WritingContinuationRequest | dict[str, Any]):
        self._ensure_started()
        return self._writing_runtime_service.continue_writing(WritingContinuationRequest.model_validate(request))

    def writing_write(self, request: WritingContinuationRequest | dict[str, Any]):
        self._ensure_started()
        return self._writing_runtime_service.write(WritingContinuationRequest.model_validate(request))

    def writing_autonomous_start(self, request: WritingContinuationRequest | dict[str, Any]):
        self._ensure_started()
        return self._writing_runtime_service.autonomous_start(WritingContinuationRequest.model_validate(request))

    def writing_autonomous_stop(self, task_id: str):
        self._ensure_started()
        return self._writing_runtime_service.autonomous_stop(task_id)

    def desktop_agent_status(self) -> dict[str, Any]:
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            return {"enabled": False, "missions": 0}
        return self._desktop_agent_runtime_service.status()

    def desktop_agent_list(self):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            return []
        return self._desktop_agent_runtime_service.list_missions()

    def desktop_agent_get(self, mission_id: str):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            raise ServiceUnavailableError("desktop agent runtime is not available")
        return self._desktop_agent_runtime_service.get_mission_status(mission_id)

    def desktop_agent_run(self, request: DesktopAgentMissionRequest | dict[str, Any]):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            raise ServiceUnavailableError("desktop agent runtime is not available")
        return self._desktop_agent_runtime_service.run(DesktopAgentMissionRequest.model_validate(request))

    def desktop_agent_pause(self, mission_id: str):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            raise ServiceUnavailableError("desktop agent runtime is not available")
        return self._desktop_agent_runtime_service.pause_mission(mission_id)

    def desktop_agent_resume(self, mission_id: str):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            raise ServiceUnavailableError("desktop agent runtime is not available")
        return self._desktop_agent_runtime_service.resume_mission(mission_id)

    def desktop_agent_abort(self, mission_id: str, *, reason: str | None = None):
        self._ensure_started()
        if self._desktop_agent_runtime_service is None:
            raise ServiceUnavailableError("desktop agent runtime is not available")
        return self._desktop_agent_runtime_service.abort_mission(mission_id, reason=reason)

    def vision_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._vision_runtime_service.status()

    def vision_capture(self, request: ScreenCaptureRequest | dict[str, Any]):
        self._ensure_started()
        parsed = ScreenCaptureRequest.model_validate(request)
        if parsed.target_type == "window":
            return self._vision_runtime_service.capture_window(parsed)
        if parsed.target_type == "region":
            return self._vision_runtime_service.capture_region(parsed)
        return self._vision_runtime_service.capture_screen(parsed)

    def vision_extract_text(self, request: OCRRequest | dict[str, Any]):
        self._ensure_started()
        return self._vision_runtime_service.extract_text(OCRRequest.model_validate(request))

    def vision_analyze(self, request: VisionAnalysisRequest | dict[str, Any]):
        self._ensure_started()
        return self._vision_runtime_service.analyze_image(VisionAnalysisRequest.model_validate(request))

    def vision_describe_active_window(self):
        self._ensure_started()
        return self._vision_runtime_service.describe_active_window()

    def vision_ui_awareness(self, request: UIAwarenessRequest | dict[str, Any]):
        self._ensure_started()
        return self._vision_runtime_service.build_ui_awareness(UIAwarenessRequest.model_validate(request))

    def vision_locate_text(self, request: TextLocationRequest | dict[str, Any]):
        self._ensure_started()
        return self._vision_runtime_service.locate_text(TextLocationRequest.model_validate(request))

    def vision_locate_element(self, request: ElementLocationRequest | dict[str, Any]):
        self._ensure_started()
        return self._vision_runtime_service.locate_element(ElementLocationRequest.model_validate(request))

    def voice_clap_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._voice_runtime_service.clap_status()

    def voice_start_session(self, request: VoiceSessionRequest | dict[str, Any]):
        self._ensure_started()
        return self._voice_runtime_service.start_session(VoiceSessionRequest.model_validate(request))

    def voice_stop_session(self):
        self._ensure_started()
        return self._voice_runtime_service.stop_session()

    def voice_transcribe_file(self, file_path: str):
        self._ensure_started()
        return self._voice_runtime_service.transcribe_file(file_path)

    def voice_speak(self, text: str):
        self._ensure_started()
        return self._voice_runtime_service.speak(text)

    def voice_dictate(self, request: VoiceSessionRequest | dict[str, Any]):
        self._ensure_started()
        return self._voice_runtime_service.dictate_once(VoiceSessionRequest.model_validate(request))

    def voice_cancel(self, correlation_id: str):
        self._ensure_started()
        return self._voice_runtime_service.cancel(correlation_id)

    def snapshot(self, *, include_history: bool = True) -> RuntimeSnapshot:
        self._ensure_started()
        return self._state_manager.snapshot(
            action_names=[definition.name for definition in self._action_router.registry.list_actions()],
            tool_names=[definition.name for definition in self._tool_registry.list_tools()],
            include_history=include_history,
        )

    def ops_status(self) -> dict[str, Any]:
        self._ensure_started()
        return self._ops_runtime_service.status()

    def ops_health(self):
        self._ensure_started()
        return self._ops_runtime_service.health()

    def ops_diagnostics(self, service_name: str | None = None):
        self._ensure_started()
        return self._ops_runtime_service.diagnostics(service_name)

    def ops_snapshot(self):
        self._ensure_started()
        return self._ops_runtime_service.snapshot()

    def ops_recover_service(self, service_name: str, *, dry_run: bool = False):
        self._ensure_started()
        return self._ops_runtime_service.recover_service(service_name, dry_run=dry_run)

    def ops_reset_breaker(self, service_name: str, dependency_name: str | None = None):
        self._ensure_started()
        return self._ops_runtime_service.reset_breaker(service_name, dependency_name)

    def ops_retention_sweep(self):
        self._ensure_started()
        return self._ops_runtime_service.retention_sweep()

    def ops_operations(self):
        self._ensure_started()
        return self._ops_runtime_service.operations()

    def ops_resources(self):
        self._ensure_started()
        return self._ops_runtime_service.resources()

    def ops_cancel_operation(self, operation_id: str, *, reason: str = "cancel requested from runtime"):
        self._ensure_started()
        return self._ops_runtime_service.cancel_operation(operation_id, reason=reason)

    def describe(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "app_name": self._settings.app_name,
            "environment": self._settings.environment,
            "mode": snapshot.mode.model_dump(mode="json"),
            "services": [entry.model_dump(mode="json") for entry in snapshot.services],
            "actions": snapshot.action_names,
            "tools": snapshot.tool_names,
            "providers": [provider.provider_name for provider in self._provider_registry.list_providers()],
            "models": self._model_service.list_models(),
            "embedding_providers": [provider.provider_name for provider in self._embedding_provider_registry.list_providers()],
            "embedding_models": self._embedding_service.list_models(),
            "semantic_memory": self._semantic_memory_service.status(),
            "indexing_runtime": self._indexing_runtime_service.status(),
            "ui_automation": self._ui_automation_service.health(),
            "vision_runtime": self._vision_runtime_service.status(),
            "voice_runtime": self._voice_runtime_service.status(),
            "system_runtime": self._system_runtime_service.status(),
            "unity_runtime": self._unity_runtime_service.status(),
            "autonomy": self._autonomy_service.status(),
            "research_runtime": self._research_runtime_service.status(),
            "science_runtime": self._science_runtime_service.status(),
            "security_runtime": self._security_runtime_service.status(),
            "self_improvement_runtime": self._self_improvement_runtime_service.status(),
            "writing_runtime": self._writing_runtime_service.status(),
            "desktop_agent_runtime": self._desktop_agent_runtime_service.status() if self._desktop_agent_runtime_service is not None else {},
            "ops_runtime": self._ops_runtime_service.status() if self._ops_runtime_service is not None else {},
            "activity_count": self._memory_service.count_activity(),
        }

    def _ensure_started(self) -> None:
        if not self._started:
            raise ServiceUnavailableError("runtime service is not started")
