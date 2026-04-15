from __future__ import annotations

import logging
from dataclasses import dataclass

from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.autonomy import (
    AutonomyService,
    MissionControlService,
    MissionExecutor,
    MissionPersistenceService,
    MissionPlanner,
    MissionReflector,
    MissionRepository,
    MissionStateManager,
    MissionVerifier,
)
from jarvis.automation.service import AutomationService
from jarvis.cognition.capabilities import install_cognitive_capabilities
from jarvis.cognition.context import RetrievedContextFormatter
from jarvis.core.metacommands import MetaCommandParser
from jarvis.core.capabilities import CapabilityRegistry
from jarvis.desktop_agent_runtime import DesktopAgentRuntimeService
from jarvis.core.diagnostics import DiagnosticsAggregator
from jarvis.core.models import AutoRecoveryPolicy, HealthStatus
from jarvis.core.modes import ModeManager
from jarvis.core.operations import AdmissionController, OperationRegistry, OperationWatchdog
from jarvis.core.process import SyncProcessRunner
from jarvis.core.resilience import CircuitBreakerPolicy, ResilienceController, ResiliencePolicy, RetryBudgetPolicy, TimeoutPolicy
from jarvis.core.lifecycle import RuntimeLifecycleSupervisor
from jarvis.core.resources import ResourceMonitor
from jarvis.core.retention import EventRetentionPolicy, LogRetentionPolicy, ReceiptRetentionPolicy, RetentionManager, SnapshotRetentionPolicy
from jarvis.core.state import RuntimeStateManager
from jarvis.core.telemetry import TelemetryRecorder
from jarvis.cognition.orchestrator import CognitiveOrchestrator
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.modules import ModuleRegistry
from jarvis.jarvis_logging import configure_logging
from jarvis.memory.repository import Database, MemoryRepository
from jarvis.memory.service import MemoryService
from jarvis.indexing_runtime import IndexingRepository, IndexingRuntimeService
from jarvis.integrations import WordCOMBackend
from jarvis.hud_runtime.service import HudRuntimeService
from jarvis.memory_semantic.embeddings import (
    EmbeddingProviderRegistry,
    EmbeddingRouter,
    EmbeddingService,
    OllamaEmbeddingProvider,
    build_default_embedding_profiles,
)
from jarvis.memory_semantic.index import RepositoryVectorIndex
from jarvis.memory_semantic.ranking import BasicHeuristicReranker, NoOpReranker
from jarvis.memory_semantic.repository import SemanticMemoryRepository
from jarvis.memory_semantic.retrieval import RetrievalPipeline
from jarvis.memory_semantic.service import SemanticMemoryService
from jarvis.models_runtime import GptOssProvider, ModelCatalog, ModelRouter, ModelService, ProviderRegistry, build_default_model_catalog
from jarvis.models_runtime.ollama import OllamaProvider
from jarvis.modules.voice_interface_module import VoiceInterfaceModule
from jarvis.modules.memory_module import MemoryModule
from jarvis.modules.operations_module import OperationsModule
from jarvis.modules.research_module import ResearchModule
from jarvis.modules.interface_module import InterfaceModule
from jarvis.modules.indexing_module import IndexingModule
from jarvis.modules.science_module import ScienceModule
from jarvis.modules.security_module import SecurityModule
from jarvis.modules.system_module import SystemModule
from jarvis.modules.unity_module import UnityModule
from jarvis.modules.autonomy_module import AutonomyModule
from jarvis.modules.vision_module import VisionModule
from jarvis.modules.voice_module import VoiceModule
from jarvis.modules.writing_module import WritingModule
from jarvis.modules.writer_module import WriterModule
from jarvis.writing_runtime import WritingAnalyzer, WritingContextResolver, WritingContinuationEngine, WritingEditor, WritingGenerator, WritingRepository, WritingRuntimeService, WritingStyleAnalyzer
from jarvis.writing_runtime.base import WritingModelAdapter
from jarvis.routing.task_router import TaskRouter
from jarvis.services.runtime import JarvisRuntimeService
from jarvis.services.ops_runtime import OpsRuntimeService
from jarvis.research_runtime import ResearchAnalyzer, ResearchPipeline, ResearchRepository, ResearchRetriever, ResearchRuntimeService, ResearchSynthesizer
from jarvis.research_runtime.base import ModelServiceResearchAdapter
from jarvis.science_runtime import ScienceRuntimeService
from jarvis.security_runtime import SecurityRuntimeService
from jarvis.self_improvement_runtime import (
    SelfImprovementAnalyzer,
    SelfImprovementExecutor,
    SelfImprovementPatchGenerator,
    SelfImprovementSandbox,
    SelfImprovementRuntimeService,
    SelfImprovementValidator,
)
from jarvis.system_runtime import (
    ApplicationCatalogRegistry,
    ApplicationResolver,
    AssociationProviderRegistry,
    AssociationResolver,
    InMemoryApplicationCatalogProvider,
    InMemoryAssociationProvider,
    InMemoryLauncherBackend,
    InMemoryVolumeProvider,
    LauncherBackendRegistry,
    NativeApplicationCatalogProvider,
    NativeAssociationProvider,
    NativeLauncherBackend,
    NativeVolumeProvider,
    PathResolver,
    ResourceSearchService,
    SystemLauncher,
    SystemRuntimeService,
    SystemTargetResolver,
    VolumeProviderRegistry,
    VolumeTopologyService,
)
from jarvis.tools.builtin import install_builtin_tools
from jarvis.unity_runtime import (
    HttpUnityBridgeBackend,
    NativeUnityInstallationProvider,
    NoOpUnityBridgeBackend,
    UnityAssetService,
    UnityBridgeRegistry,
    UnityBridgeService,
    UnityEditorSessionRegistry,
    UnityEditorOperationService,
    UnityInstallationDiscoveryService,
    UnityInstallationRegistry,
    UnityLaunchIntegrationService,
    UnityProjectDiscoveryService,
    UnityProjectResolver,
    UnityProjectService,
    UnityRuntimeService,
    UnityScriptService,
)
from jarvis.tools.registry import ToolRegistry
from jarvis.ui_automation import InMemoryDesktopAutomationBackend, UIAutomationService, WindowsDesktopAutomationBackend
from jarvis.ui_automation.base import CancellationRequest, WriteTextRequest
from jarvis.vision_runtime import (
    HeuristicVisionAnalyzer,
    InMemoryOCRProvider,
    InMemoryScreenCaptureBackend,
    MssScreenCaptureBackend,
    OCRProviderRegistry,
    ScreenCaptureBackendRegistry,
    TesseractOCRProvider,
    VisionAnalyzerRegistry,
    VisionRuntimeService,
    WindowsScreenCaptureBackend,
    WindowsUIAwarenessBackend,
)
from jarvis.voice_runtime import VoiceRuntimeService
from jarvis.voice_runtime.backends import (
    AudioInputRegistry,
    AudioOutputRegistry,
    FasterWhisperSTTProvider,
    InMemoryAudioInputBackend,
    InMemoryAudioOutputBackend,
    InMemorySTTProvider,
    InMemoryTTSProvider,
    Pyttsx3AudioOutputBackend,
    Pyttsx3TTSProvider,
    SoundDeviceAudioInputBackend,
    STTProviderRegistry,
    TTSProviderRegistry,
    WavFileAudioInputBackend,
    WinsoundAudioOutputBackend,
)
from jarvis.voice_runtime.coqui_tts import CoquiXTTSProvider
from jarvis.voice_runtime.stt import STTService
from jarvis.voice_runtime.tts import TTSService


@dataclass
class JarvisApplication:
    settings: Settings
    event_bus: EventBus
    mode_manager: ModeManager
    state_manager: RuntimeStateManager
    capability_registry: CapabilityRegistry
    action_registry: ActionRegistry
    action_router: ActionRouter
    tool_registry: ToolRegistry
    provider_registry: ProviderRegistry
    model_catalog: ModelCatalog
    model_router: ModelRouter
    model_service: ModelService
    embedding_provider_registry: EmbeddingProviderRegistry
    embedding_router: EmbeddingRouter
    embedding_service: EmbeddingService
    ui_automation_service: UIAutomationService
    vision_runtime_service: VisionRuntimeService
    voice_runtime_service: VoiceRuntimeService
    system_runtime_service: SystemRuntimeService
    unity_runtime_service: UnityRuntimeService
    autonomy_service_runtime: AutonomyService
    research_runtime_service: ResearchRuntimeService
    science_runtime_service: ScienceRuntimeService
    security_runtime_service: SecurityRuntimeService
    self_improvement_runtime_service: SelfImprovementRuntimeService
    writing_runtime_service: WritingRuntimeService
    desktop_agent_runtime_service: DesktopAgentRuntimeService
    meta_command_parser: MetaCommandParser
    task_router: TaskRouter
    memory_service: MemoryService
    semantic_memory_service: SemanticMemoryService
    indexing_runtime_service: IndexingRuntimeService
    hud_runtime_service: HudRuntimeService
    mission_persistence_service: MissionPersistenceService
    module_registry: ModuleRegistry
    automation_service: AutomationService
    orchestrator: CognitiveOrchestrator
    runtime_service: JarvisRuntimeService
    ops_runtime_service: OpsRuntimeService
    lifecycle_supervisor: RuntimeLifecycleSupervisor
    logger: logging.Logger
    started: bool = False

    def start(self) -> None:
        if self.started:
            return
        self.settings.prepare_environment()
        self.state_manager.update_service("jarvis.app", HealthStatus.STARTING)
        self.memory_service.create_schema()
        self.state_manager.update_service("memory", HealthStatus.READY)
        self.semantic_memory_service.create_schema()
        self.state_manager.update_service("semantic_memory", HealthStatus.READY)
        self.indexing_runtime_service.create_schema()
        self.state_manager.update_service("indexing_runtime", HealthStatus.READY, self.indexing_runtime_service.status())
        self.mission_persistence_service.create_schema()
        self.mission_persistence_service.hydrate_state(self.autonomy_service_runtime._state)  # noqa: SLF001
        self.lifecycle_supervisor.start_service("system_runtime")
        self.state_manager.update_service("ui_automation", HealthStatus.READY, self.ui_automation_service.health())
        self.state_manager.update_service("vision_runtime", HealthStatus.READY, self.vision_runtime_service.status())
        self.state_manager.update_service("voice_runtime", HealthStatus.READY, self.voice_runtime_service.status())
        self.state_manager.update_service("system_runtime", HealthStatus.READY, self.system_runtime_service.status())
        self.lifecycle_supervisor.start_service("unity_runtime")
        self.state_manager.update_service("unity_runtime", HealthStatus.READY, self.unity_runtime_service.status())
        self.state_manager.update_service("autonomy", HealthStatus.READY, self.autonomy_service_runtime.status())
        self.research_runtime_service.start()
        self.state_manager.update_service("research_runtime", HealthStatus.READY, self.research_runtime_service.status())
        self.science_runtime_service.start()
        self.state_manager.update_service("science_runtime", HealthStatus.READY, self.science_runtime_service.status())
        self.security_runtime_service.start()
        self.state_manager.update_service("security_runtime", HealthStatus.READY, self.security_runtime_service.status())
        self.self_improvement_runtime_service.start()
        self.state_manager.update_service("self_improvement_runtime", HealthStatus.READY, self.self_improvement_runtime_service.status())
        self.writing_runtime_service.start()
        self.state_manager.update_service("writing_runtime", HealthStatus.READY, self.writing_runtime_service.status())
        self.desktop_agent_runtime_service.start()
        self.state_manager.update_service("desktop_agent_runtime", HealthStatus.READY, self.desktop_agent_runtime_service.status())
        self.indexing_runtime_service.start()
        self.state_manager.update_service("indexing_runtime", HealthStatus.READY, self.indexing_runtime_service.status())
        self.hud_runtime_service.start()
        self.state_manager.update_service("hud_runtime", HealthStatus.READY, self.hud_runtime_service.status())
        self.module_registry.start_all()
        self.state_manager.update_service("modules", HealthStatus.READY, {"count": len(self.module_registry.descriptors)})
        self.state_manager.update_service(
            "providers",
            HealthStatus.READY,
            {"providers": [provider.provider_name for provider in self.provider_registry.list_providers()]},
        )
        self.state_manager.update_service(
            "models",
            HealthStatus.READY,
            {"profiles": [profile["logical_name"] for profile in self.model_service.list_models()]},
        )
        self.state_manager.update_service(
            "embedding_providers",
            HealthStatus.READY,
            {"providers": [provider.provider_name for provider in self.embedding_provider_registry.list_providers()]},
        )
        self.state_manager.update_service(
            "semantic_embeddings",
            HealthStatus.READY,
            {"profiles": [profile["logical_name"] for profile in self.embedding_service.list_models()]},
        )
        self.lifecycle_supervisor.start_service("automation")
        self.state_manager.update_service("automation", HealthStatus.READY)
        self.state_manager.update_service("task_router", HealthStatus.READY)
        self.lifecycle_supervisor.start_service("runtime")
        self.lifecycle_supervisor.start_service("ops_runtime")
        self.state_manager.update_service("runtime", HealthStatus.READY, self.runtime_service.health().details)
        self.state_manager.update_service("ops_runtime", HealthStatus.READY, self.ops_runtime_service.status())
        self.started = True
        self.state_manager.update_service("jarvis.app", HealthStatus.READY)
        self.logger.info("jarvis_started", extra={"modules": [item.name for item in self.module_registry.descriptors]})

    def stop(self) -> None:
        if not self.started:
            return
        self.state_manager.update_service("jarvis.app", HealthStatus.STOPPED)
        self.lifecycle_supervisor.stop_service("ops_runtime")
        self.lifecycle_supervisor.stop_service("runtime")
        self.lifecycle_supervisor.stop_service("automation")
        self.lifecycle_supervisor.stop_service("system_runtime")
        self.lifecycle_supervisor.stop_service("unity_runtime")
        self.state_manager.update_service("automation", HealthStatus.STOPPED)
        self.module_registry.stop_all()
        self.state_manager.update_service("modules", HealthStatus.STOPPED, {"count": len(self.module_registry.descriptors)})
        self.state_manager.update_service("providers", HealthStatus.STOPPED)
        self.state_manager.update_service("models", HealthStatus.STOPPED)
        self.state_manager.update_service("embedding_providers", HealthStatus.STOPPED)
        self.state_manager.update_service("semantic_embeddings", HealthStatus.STOPPED)
        self.state_manager.update_service("task_router", HealthStatus.STOPPED)
        self.state_manager.update_service("runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("ops_runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("semantic_memory", HealthStatus.STOPPED)
        self.state_manager.update_service("ui_automation", HealthStatus.STOPPED)
        self.state_manager.update_service("vision_runtime", HealthStatus.STOPPED)
        self.voice_runtime_service.stop()
        self.state_manager.update_service("voice_runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("system_runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("unity_runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("autonomy", HealthStatus.STOPPED)
        self.research_runtime_service.stop()
        self.state_manager.update_service("research_runtime", HealthStatus.STOPPED)
        self.science_runtime_service.stop()
        self.state_manager.update_service("science_runtime", HealthStatus.STOPPED)
        self.security_runtime_service.stop()
        self.state_manager.update_service("security_runtime", HealthStatus.STOPPED)
        self.self_improvement_runtime_service.stop()
        self.state_manager.update_service("self_improvement_runtime", HealthStatus.STOPPED)
        self.writing_runtime_service.stop()
        self.state_manager.update_service("writing_runtime", HealthStatus.STOPPED)
        self.desktop_agent_runtime_service.stop()
        self.state_manager.update_service("desktop_agent_runtime", HealthStatus.STOPPED)
        self.indexing_runtime_service.stop()
        self.state_manager.update_service("indexing_runtime", HealthStatus.STOPPED)
        self.hud_runtime_service.stop()
        self.state_manager.update_service("hud_runtime", HealthStatus.STOPPED)
        self.state_manager.update_service("memory", HealthStatus.STOPPED)
        self.started = False
        self.logger.info("jarvis_stopped")

    def describe(self) -> dict[str, object]:
        runtime_description = self.runtime_service.describe()
        return {
            "app_name": self.settings.app_name,
            "environment": self.settings.environment,
            "mode": runtime_description["mode"],
            "services": runtime_description["services"],
            "modules": [{"name": item.name, "description": item.description} for item in self.module_registry.descriptors],
            "actions": [
                {
                    "name": definition.name,
                    "description": definition.description,
                    "tags": list(definition.tags),
                }
                for definition in self.action_registry.list_actions()
            ],
            "tools": [
                {
                    "name": definition.name,
                    "description": definition.description,
                    "tags": list(definition.tags),
                }
                for definition in self.tool_registry.list_tools()
            ],
            "capabilities": [
                capability.descriptor.model_dump(mode="json")
                for capability in self.capability_registry.list_capabilities()
            ],
            "providers": [health.model_dump(mode="json") for health in self.model_service.health()],
            "models": self.model_service.list_models(),
            "embedding_providers": [health.model_dump(mode="json") for health in self.embedding_service.health()],
            "embedding_models": self.embedding_service.list_models(),
            "semantic_memory": self.semantic_memory_service.status(),
            "indexing_runtime": self.indexing_runtime_service.status(),
            "hud_runtime": self.hud_runtime_service.status(),
            "ui_automation": self.ui_automation_service.health(),
            "vision_runtime": self.vision_runtime_service.status(),
            "voice_runtime": self.voice_runtime_service.status(),
            "system_runtime": self.system_runtime_service.status(),
            "unity_runtime": self.unity_runtime_service.status(),
            "autonomy": self.autonomy_service_runtime.status(),
            "research_runtime": self.research_runtime_service.status(),
            "self_improvement_runtime": self.self_improvement_runtime_service.status(),
            "writing_runtime": self.writing_runtime_service.status(),
            "desktop_agent_runtime": self.desktop_agent_runtime_service.status(),
            "ops_runtime": self.ops_runtime_service.status(),
            "activity_count": self.memory_service.count_activity(),
        }


def build_application(settings: Settings | None = None) -> JarvisApplication:
    settings = settings or Settings()
    settings.prepare_environment()
    configure_logging(
        settings.log_level,
        settings.json_logs,
        log_file=settings.resolved_log_file,
        max_bytes=settings.log_file_max_bytes,
        backup_count=settings.log_file_backup_count,
    )
    logger = logging.getLogger("jarvis.app")
    database = Database(settings.sqlalchemy_database_url, settings.sqlalchemy_engine_options())
    repository = MemoryRepository(database)
    semantic_repository = SemanticMemoryRepository(database)
    indexing_repository = IndexingRepository(database)
    mission_repository = MissionRepository(database)
    research_repository = ResearchRepository(database)
    writing_repository = WritingRepository(database)
    memory_service = MemoryService(repository)
    event_bus = EventBus(history_limit=settings.ops_event_history_limit)
    mode_manager = ModeManager(settings.default_mode)
    state_manager = RuntimeStateManager(
        settings.app_name,
        settings.environment,
        mode_manager,
        history_limit=settings.state_history_limit,
    )
    state_manager.bind(event_bus)
    telemetry = TelemetryRecorder(history_limit=settings.state_history_limit)
    admission = AdmissionController(
        default_limit=settings.ops_default_max_concurrent_operations,
        queue_limit=settings.ops_default_queue_limit,
        policies={
            "autonomy": {"max_concurrent": settings.autonomy_max_concurrent_missions, "queue_limit": settings.autonomy_max_concurrent_missions, "overflow_policy": "reject"},
            "research_runtime": {"max_concurrent": settings.research_max_concurrent_tasks, "queue_limit": settings.research_max_concurrent_tasks, "overflow_policy": "defer"},
            "indexing_runtime": {"max_concurrent": settings.indexing_max_concurrent_runs, "queue_limit": settings.indexing_max_concurrent_runs, "overflow_policy": "reject"},
            "ui_automation": {"max_concurrent": 1, "queue_limit": 1, "overflow_policy": "reject"},
            "voice_runtime": {"max_concurrent": 2, "queue_limit": 2, "overflow_policy": "reject"},
            "vision_runtime": {"max_concurrent": 3, "queue_limit": 4, "overflow_policy": "reject"},
            "system_runtime": {"max_concurrent": 4, "queue_limit": 6, "overflow_policy": "reject"},
            "unity_runtime": {"max_concurrent": 2, "queue_limit": 2, "overflow_policy": "reject"},
            "writing_runtime": {"max_concurrent": 3, "queue_limit": 4, "overflow_policy": "defer"},
        },
    )
    operation_registry = OperationRegistry(event_bus=event_bus, admission_controller=admission, history_limit=settings.ops_operation_history_limit)
    resilience = ResilienceController(
        telemetry,
        default_policy=ResiliencePolicy(
            circuit_breaker=CircuitBreakerPolicy(
                failure_threshold=settings.ops_breaker_failure_threshold,
                recovery_timeout_seconds=settings.ops_breaker_recovery_timeout_seconds,
            ),
            retry_budget=RetryBudgetPolicy(
                max_attempts=settings.ops_retry_max_attempts,
                max_retries_per_window=settings.ops_retry_budget_max_retries,
                window_seconds=settings.ops_retry_budget_window_seconds,
                base_backoff_seconds=settings.ops_retry_base_backoff_seconds,
                max_backoff_seconds=settings.ops_retry_max_backoff_seconds,
            ),
            timeout=TimeoutPolicy(
                timeout_ms=settings.ops_default_timeout_ms,
                health_timeout_ms=settings.ops_health_timeout_ms,
                recovery_timeout_ms=settings.ops_recovery_timeout_ms,
                shutdown_timeout_ms=settings.ops_shutdown_timeout_ms,
            ),
            slow_operation_threshold_ms=settings.ops_slow_operation_threshold_ms,
        ),
        event_bus=event_bus,
    )
    lifecycle_supervisor = RuntimeLifecycleSupervisor(telemetry, event_bus=event_bus)
    resource_monitor = ResourceMonitor(
        workspace_root=settings.resolved_workspace_root,
        data_dir=settings.resolved_data_dir,
        history_limit=settings.ops_operation_history_limit,
        poll_interval_seconds=settings.ops_resource_poll_interval_seconds,
        event_bus=event_bus,
    )
    capability_registry = CapabilityRegistry()
    provider_registry = ProviderRegistry()
    if settings.ollama_enabled:
        provider_registry.register(OllamaProvider(settings))
    if settings.gpt_oss_enabled or settings.general_chat_model_provider == "gpt_oss":
        provider_registry.register(GptOssProvider(settings))
    model_catalog = build_default_model_catalog(settings)
    model_router = ModelRouter(model_catalog, mode_manager)
    model_service = ModelService(settings, mode_manager, provider_registry, model_catalog, model_router, event_bus, logger=logger, resilience_controller=resilience)
    embedding_provider_registry = EmbeddingProviderRegistry()
    if settings.embeddings_enabled:
        embedding_provider_registry.register(OllamaEmbeddingProvider(settings))
    embedding_router = EmbeddingRouter(build_default_embedding_profiles(settings), mode_manager)
    embedding_service = EmbeddingService(
        settings,
        mode_manager,
        embedding_provider_registry,
        embedding_router,
        event_bus,
        logger=logger,
        resilience_controller=resilience,
    )
    vector_index = RepositoryVectorIndex(semantic_repository)
    reranker = BasicHeuristicReranker() if settings.semantic_reranking_type == "basic" else NoOpReranker()
    retrieval_pipeline = RetrievalPipeline(
        semantic_repository,
        vector_index,
        reranker,
        embedding_service,
        logger=logger,
    )
    semantic_memory_service = SemanticMemoryService(
        settings,
        semantic_repository,
        embedding_service,
        vector_index,
        retrieval_pipeline,
        memory_service,
        logger=logger,
    )
    indexing_runtime_service = IndexingRuntimeService(
        settings,
        event_bus,
        indexing_repository,
        semantic_memory_service,
        embedding_service,
        system_runtime_service=None,
        research_repository=research_repository,
        writing_repository=writing_repository,
        telemetry=telemetry,
        logger=logger,
    )
    action_registry = ActionRegistry()
    router = ActionRouter(action_registry, settings, memory_service, model_service, event_bus, logger=logger)
    tool_registry = ToolRegistry(settings, memory_service, router, model_service, event_bus, logger=logger)
    meta_command_parser = MetaCommandParser()
    process_runner = SyncProcessRunner()
    application_registry = ApplicationCatalogRegistry()
    if settings.ui_backend_kind == "in_memory":
        ui_backend = InMemoryDesktopAutomationBackend()
    else:
        ui_backend = WindowsDesktopAutomationBackend()
    if settings.system_backend_kind == "in_memory":
        application_registry.register(InMemoryApplicationCatalogProvider())
    else:
        application_registry.register(NativeApplicationCatalogProvider(settings.resolved_workspace_root, settings.system_known_locations))
    ui_automation_service = UIAutomationService(
        settings,
        mode_manager,
        ui_backend,
        event_bus,
        logger=logger,
        operation_registry=operation_registry,
        application_registry=application_registry,
    )
    capture_registry = ScreenCaptureBackendRegistry()
    capture_registry.register(InMemoryScreenCaptureBackend(ui_backend))
    if settings.ui_backend_kind != "in_memory":
        capture_registry.register(MssScreenCaptureBackend(ui_backend))
        capture_registry.register(WindowsScreenCaptureBackend(ui_backend))
    ocr_registry = OCRProviderRegistry()
    ocr_registry.register(InMemoryOCRProvider())
    if settings.vision_ocr_provider_default == "tesseract_ocr" or "tesseract_ocr" in settings.vision_ocr_provider_fallback_order:
        ocr_registry.register(TesseractOCRProvider())
    analyzer_registry = VisionAnalyzerRegistry()
    analyzer_registry.register(HeuristicVisionAnalyzer())
    analyzer_registry.register(WindowsUIAwarenessBackend(ui_backend))
    vision_runtime_service = VisionRuntimeService(
        settings,
        mode_manager,
        event_bus,
        capture_registry,
        ocr_registry,
        analyzer_registry,
        ui_metadata_adapter=ui_backend,
        logger=logger,
        resilience_controller=resilience,
        operation_registry=operation_registry,
    )
    audio_input_registry = AudioInputRegistry()
    audio_input_registry.register(WavFileAudioInputBackend())
    audio_input_registry.register(InMemoryAudioInputBackend())
    if settings.voice_audio_input_backend_default == "sounddevice" or settings.voice_input_enabled:
        audio_input_registry.register(SoundDeviceAudioInputBackend())
    audio_output_registry = AudioOutputRegistry()
    audio_output_registry.register(InMemoryAudioOutputBackend())
    audio_output_registry.register(WinsoundAudioOutputBackend())
    if settings.voice_audio_output_backend_default == "pyttsx3" or settings.voice_tts_provider_default == "pyttsx3" or "pyttsx3" in settings.voice_tts_provider_fallback_order:
        audio_output_registry.register(Pyttsx3AudioOutputBackend())
    stt_registry = STTProviderRegistry()
    stt_registry.register(InMemorySTTProvider())
    if (
        settings.voice_stt_provider_default == "faster_whisper"
        or "faster_whisper" in settings.voice_stt_provider_fallback_order
        or settings.voice_input_provider_default == "faster_whisper"
    ):
        stt_registry.register(FasterWhisperSTTProvider())
    tts_registry = TTSProviderRegistry()
    tts_registry.register(InMemoryTTSProvider())
    tts_registry.register(
        CoquiXTTSProvider(
            model_name=settings.voice_coqui_model_name,
            language=settings.voice_default_language,
            speaker_wav=settings.resolved_voice_coqui_speaker_wav,
            speaker=settings.voice_coqui_speaker_name,
            device_preference=settings.voice_coqui_device_preference,
            tos_agreed=settings.voice_coqui_tos_agreed,
            logger=logger,
        )
    )
    if settings.voice_tts_provider_default == "pyttsx3" or "pyttsx3" in settings.voice_tts_provider_fallback_order:
        tts_registry.register(Pyttsx3TTSProvider())
    stt_service = STTService(settings, mode_manager, stt_registry, event_bus, logger=logger, resilience_controller=resilience)
    tts_service = TTSService(settings, mode_manager, tts_registry, event_bus, logger=logger, resilience_controller=resilience)
    voice_runtime_service = VoiceRuntimeService(
        settings,
        mode_manager,
        event_bus,
        audio_input_registry,
        audio_output_registry,
        stt_service,
        tts_service,
        logger=logger,
        operation_registry=operation_registry,
        dictate_callback=lambda text, correlation_id, options: ui_automation_service.write_text(
            WriteTextRequest.model_validate(
                {
                    "text": text,
                    "mode": options.get("ui_mode") or "copilot",
                    "target_window": options.get("target_window"),
                    "typing_interval_ms": 0,
                    "pause_between_blocks_ms": 0,
                    "approved": True,
                }
            ),
            correlation_id=correlation_id,
        ),
        cancel_callback=lambda correlation_id: ui_automation_service.cancel(CancellationRequest(correlation_id=correlation_id)),
    )
    voice_runtime_service.set_voice_enabled(settings.voice_enabled)
    voice_runtime_service.set_voice_muted(settings.voice_start_muted)
    volume_registry = VolumeProviderRegistry()
    association_registry = AssociationProviderRegistry()
    launcher_registry = LauncherBackendRegistry()
    if settings.system_backend_kind == "in_memory":
        volume_registry.register(InMemoryVolumeProvider())
        association_registry.register(InMemoryAssociationProvider())
        launcher_registry.register(InMemoryLauncherBackend())
        launcher_backend_name = "in_memory_launcher"
    else:
        volume_registry.register(NativeVolumeProvider(settings.resolved_workspace_root, settings.system_known_locations))
        association_registry.register(NativeAssociationProvider())
        launcher_registry.register(NativeLauncherBackend())
        launcher_backend_name = "native_launcher"
    volume_topology = VolumeTopologyService(volume_registry, settings, logger=logger)
    resource_search = ResourceSearchService(settings, volume_topology, application_registry, logger=logger)
    application_resolver = ApplicationResolver(application_registry, logger=logger)
    path_resolver = PathResolver()
    association_resolver = AssociationResolver(association_registry, logger=logger)
    system_launcher = SystemLauncher(launcher_registry, backend_name=launcher_backend_name, logger=logger)
    system_target_resolver = SystemTargetResolver(application_resolver, path_resolver, resource_search, association_resolver, logger=logger)
    system_runtime_service = SystemRuntimeService(
        settings,
        event_bus,
        volume_topology,
        resource_search,
        system_target_resolver,
        system_launcher,
        logger=logger,
        operation_registry=operation_registry,
    )
    indexing_runtime_service._system_runtime = system_runtime_service  # noqa: SLF001
    unity_installation_registry = UnityInstallationRegistry()
    unity_installation_registry.register(
        NativeUnityInstallationProvider(
            known_locations=settings.unity_known_locations,
            fallback_paths=settings.unity_editor_fallback_paths,
        )
    )
    unity_installation_discovery = UnityInstallationDiscoveryService(unity_installation_registry, settings, logger=logger)
    unity_bridge_registry = UnityBridgeRegistry()
    unity_bridge_registry.register(NoOpUnityBridgeBackend())
    unity_bridge_registry.register(HttpUnityBridgeBackend(settings, logger=logger, resilience_controller=resilience))
    unity_session_registry = UnityEditorSessionRegistry()
    unity_project_discovery = UnityProjectDiscoveryService(settings, logger=logger)
    unity_project_service = UnityProjectService(logger=logger)
    unity_project_resolver = UnityProjectResolver(unity_project_discovery, unity_project_service, logger=logger)
    unity_asset_service = UnityAssetService(logger=logger)
    unity_script_service = UnityScriptService(settings, logger=logger)
    unity_editor_ops = UnityEditorOperationService(logger=logger)
    unity_launch_service = UnityLaunchIntegrationService(settings, system_runtime_service, logger=logger)
    unity_bridge_service = UnityBridgeService(
        unity_bridge_registry,
        backend_name=settings.unity_bridge_backend_default,
        session_registry=unity_session_registry,
        logger=logger,
    )
    unity_runtime_service = UnityRuntimeService(
        settings,
        event_bus,
        unity_installation_discovery,
        unity_project_discovery,
        unity_project_resolver,
        unity_project_service,
        unity_asset_service,
        unity_script_service,
        unity_editor_ops,
        unity_bridge_service,
        unity_launch_service,
        unity_session_registry,
        logger=logger,
        operation_registry=operation_registry,
    )
    autonomy_state_manager = MissionStateManager()
    mission_persistence = MissionPersistenceService(mission_repository, logger=logger)
    autonomy_planner = MissionPlanner(model_service, settings, logger=logger)
    autonomy_executor = MissionExecutor(
        runtime_service=None,
        semantic_memory=semantic_memory_service,
        ui_automation=ui_automation_service,
        voice_runtime=voice_runtime_service,
        vision_runtime=vision_runtime_service,
        model_service=model_service,
        logger=logger,
        operation_registry=operation_registry,
    )
    autonomy_verifier = MissionVerifier(vision_runtime=vision_runtime_service, ui_automation=ui_automation_service, logger=logger)
    autonomy_reflector = MissionReflector(logger=logger)
    autonomy_control = MissionControlService(mission_persistence, autonomy_state_manager, logger=logger)
    autonomy_service_runtime = AutonomyService(
        settings,
        mode_manager,
        event_bus,
        autonomy_planner,
        autonomy_executor,
        autonomy_verifier,
        autonomy_reflector,
        autonomy_state_manager,
        autonomy_control,
        mission_persistence,
        logger=logger,
        operation_registry=operation_registry,
    )
    research_repository.create_schema()
    research_model_adapter = ModelServiceResearchAdapter(model_service)
    research_retriever = ResearchRetriever(settings, semantic_memory_service, memory_service, vision_runtime_service, system_runtime_service, logger=logger)
    research_analyzer = ResearchAnalyzer(research_model_adapter)
    research_synthesizer = ResearchSynthesizer(research_model_adapter)
    research_pipeline = ResearchPipeline(
        retriever=research_retriever,
        analyzer=research_analyzer,
        synthesizer=research_synthesizer,
        semantic_memory=semantic_memory_service,
        logger=logger,
    )
    research_runtime_service = ResearchRuntimeService(
        settings,
        event_bus,
        research_repository,
        research_pipeline,
        autonomy_service=autonomy_service_runtime,
        logger=logger,
        operation_registry=operation_registry,
    )
    science_runtime_service = ScienceRuntimeService(settings, logger=logger)
    security_runtime_service = SecurityRuntimeService(settings, logger=logger)
    writing_repository.create_schema()
    writing_model_adapter = WritingModelAdapter(model_service)
    self_improvement_analyzer = SelfImprovementAnalyzer(
        ops_runtime=None,
        research_runtime=research_runtime_service,
        log_file=settings.resolved_log_file,
        logger=logger,
    )
    self_improvement_patch_generator = SelfImprovementPatchGenerator(model_adapter=writing_model_adapter, logger=logger)
    self_improvement_sandbox = SelfImprovementSandbox(data_dir=settings.resolved_data_dir, logger=logger)
    self_improvement_validator = SelfImprovementValidator(logger=logger)
    self_improvement_executor = SelfImprovementExecutor(logger=logger)
    self_improvement_runtime_service = SelfImprovementRuntimeService(
        settings=settings,
        analyzer=self_improvement_analyzer,
        patch_generator=self_improvement_patch_generator,
        sandbox=self_improvement_sandbox,
        validator=self_improvement_validator,
        executor=self_improvement_executor,
        logger=logger,
        operation_registry=operation_registry,
    )
    word_backend = None if settings.ui_backend_kind == "in_memory" else WordCOMBackend(logger=logger)
    writing_context_resolver = WritingContextResolver(semantic_memory_service, vision_runtime_service, ui_automation_service, word_backend=word_backend, logger=logger)
    writing_style_analyzer = WritingStyleAnalyzer(writing_model_adapter)
    writing_analyzer = WritingAnalyzer()
    writing_generator = WritingGenerator(writing_model_adapter, settings)
    writing_continuation = WritingContinuationEngine(writing_generator)
    writing_editor = WritingEditor(ui_automation_service, settings, word_backend=word_backend, logger=logger)
    writing_runtime_service = WritingRuntimeService(
        settings,
        event_bus,
        writing_repository,
        writing_context_resolver,
        writing_style_analyzer,
        writing_analyzer,
        writing_continuation,
        writing_editor,
        semantic_memory_service,
        ui_automation_service,
        autonomy_service=autonomy_service_runtime,
        logger=logger,
        operation_registry=operation_registry,
    )

    module_registry = ModuleRegistry()
    module_registry.register(MemoryModule(memory_service))
    module_registry.register(IndexingModule(indexing_runtime_service))
    module_registry.register(ResearchModule(research_runtime_service))
    module_registry.register(ScienceModule(science_runtime_service))
    module_registry.register(SecurityModule(security_runtime_service))
    module_registry.register(WriterModule())
    module_registry.register(WritingModule(writing_runtime_service))
    module_registry.register(InterfaceModule(ui_automation_service, semantic_memory_service))
    module_registry.register(VisionModule(vision_runtime_service))
    module_registry.register(VoiceModule())
    module_registry.register(VoiceInterfaceModule(voice_runtime_service))
    module_registry.register(SystemModule(system_runtime_service))
    module_registry.register(UnityModule(unity_runtime_service))
    module_registry.register(AutonomyModule(autonomy_service_runtime))
    module_registry.register(OperationsModule(process_runner))
    module_registry.register_actions(action_registry)
    module_registry.register_tools(tool_registry)
    module_registry.register_capabilities(capability_registry)
    install_cognitive_capabilities(capability_registry)

    automation_service = AutomationService(
        memory_service,
        router,
        action_registry,
        timezone_name=settings.automation_timezone,
        logger=logger,
    )
    orchestrator = CognitiveOrchestrator(
        router,
        memory_service,
        capability_registry,
        model_service,
        semantic_memory_service,
        RetrievedContextFormatter(char_budget=settings.semantic_context_char_budget),
        settings.semantic_context_max_chunks,
        ui_automation_service,
        voice_runtime_service,
    )
    task_router = TaskRouter(
        mode_manager=mode_manager,
        state_manager=state_manager,
        meta_command_parser=meta_command_parser,
        capability_registry=capability_registry,
        action_router=router,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
    )
    install_builtin_tools(
        tool_registry,
        state_provider=lambda include_history=True: state_manager.snapshot(
            action_names=[definition.name for definition in action_registry.list_actions()],
            tool_names=[definition.name for definition in tool_registry.list_tools()],
            include_history=include_history,
        ),
    )
    runtime_service = JarvisRuntimeService(
        settings=settings,
        mode_manager=mode_manager,
        state_manager=state_manager,
        task_router=task_router,
        action_router=router,
        tool_registry=tool_registry,
        provider_registry=provider_registry,
        model_catalog=model_catalog,
        model_router=model_router,
        model_service=model_service,
        embedding_provider_registry=embedding_provider_registry,
        embedding_router=embedding_router,
        embedding_service=embedding_service,
        ui_automation_service=ui_automation_service,
        vision_runtime_service=vision_runtime_service,
        voice_runtime_service=voice_runtime_service,
        system_runtime_service=system_runtime_service,
        unity_runtime_service=unity_runtime_service,
        autonomy_service=autonomy_service_runtime,
        research_runtime_service=research_runtime_service,
        science_runtime_service=science_runtime_service,
        security_runtime_service=security_runtime_service,
        self_improvement_runtime_service=self_improvement_runtime_service,
        writing_runtime_service=writing_runtime_service,
        memory_service=memory_service,
        semantic_memory_service=semantic_memory_service,
        indexing_runtime_service=indexing_runtime_service,
        automation_service=automation_service,
        ops_runtime_service=None,
        desktop_agent_runtime_service=None,
    )
    desktop_agent_runtime_service = DesktopAgentRuntimeService(
        settings=settings,
        runtime=runtime_service,
        ui_backend=ui_backend,
        logger=logger,
    )
    runtime_service._desktop_agent_runtime_service = desktop_agent_runtime_service  # noqa: SLF001
    retention_manager = RetentionManager(
        state_manager=state_manager,
        event_bus=event_bus,
        telemetry=telemetry,
        log_directory=settings.resolved_logs_dir,
        receipt_policy=ReceiptRetentionPolicy(max_count=settings.ops_receipt_retention_limit, max_age_seconds=settings.ops_retention_max_age_seconds),
        event_policy=EventRetentionPolicy(max_count=settings.ops_event_history_limit, max_age_seconds=settings.ops_retention_max_age_seconds),
        snapshot_policy=SnapshotRetentionPolicy(max_count=settings.ops_snapshot_history_limit, max_age_seconds=settings.ops_snapshot_retention_max_age_seconds),
        log_policy=LogRetentionPolicy(
            max_files=settings.log_file_backup_count,
            max_total_bytes=settings.log_file_max_bytes * settings.log_file_backup_count,
            max_age_seconds=settings.ops_log_retention_max_age_seconds,
        ),
    )
    operation_watchdog = OperationWatchdog(
        operation_registry,
        event_bus=event_bus,
        poll_interval_seconds=settings.ops_watchdog_poll_interval_seconds,
        hard_timeout_callback=lambda record: lifecycle_supervisor.attempt_auto_recover(record.service_name, reason=f"watchdog timeout: {record.operation_name}")
        if record.service_name in {"system_runtime", "unity_runtime"}
        else None,
    )
    ops_runtime_service = OpsRuntimeService(
        settings=settings,
        mode_manager=mode_manager,
        state_manager=state_manager,
        event_bus=event_bus,
        telemetry=telemetry,
        resilience_controller=resilience,
        lifecycle_supervisor=lifecycle_supervisor,
        retention_manager=retention_manager,
        operation_registry=operation_registry,
        operation_watchdog=operation_watchdog,
        resource_monitor=resource_monitor,
        admission_controller=admission,
    )
    hud_runtime_service = HudRuntimeService(
        settings=settings,
        event_bus=event_bus,
        state_manager=state_manager,
        runtime_service=runtime_service,
        ops_runtime=ops_runtime_service,
        autonomy_service=autonomy_service_runtime,
        research_runtime=research_runtime_service,
        writing_runtime=writing_runtime_service,
        indexing_runtime=indexing_runtime_service,
        unity_runtime=unity_runtime_service,
        system_runtime=system_runtime_service,
        vision_runtime=vision_runtime_service,
        voice_runtime=voice_runtime_service,
        logger=logger,
    )
    runtime_service._ops_runtime_service = ops_runtime_service  # noqa: SLF001
    self_improvement_analyzer._ops_runtime = ops_runtime_service  # noqa: SLF001
    lifecycle_supervisor.register("system_runtime", start=system_runtime_service.start, stop=system_runtime_service.stop, health=system_runtime_service.health)
    lifecycle_supervisor.register("unity_runtime", start=unity_runtime_service.start, stop=unity_runtime_service.stop, health=unity_runtime_service.health)
    lifecycle_supervisor.register("automation", start=automation_service.start, stop=automation_service.stop)
    lifecycle_supervisor.register("runtime", start=runtime_service.start, stop=runtime_service.stop, health=runtime_service.health)
    lifecycle_supervisor.register("ops_runtime", start=ops_runtime_service.start, stop=ops_runtime_service.stop)
    lifecycle_supervisor.configure_auto_recovery(
        "system_runtime",
        AutoRecoveryPolicy(
            enabled=settings.ops_auto_recover_system,
            cooldown_seconds=settings.ops_auto_recover_cooldown_seconds,
            max_attempts_per_window=settings.ops_auto_recover_max_attempts_per_window,
            window_seconds=settings.ops_auto_recover_window_seconds,
        ),
    )
    lifecycle_supervisor.configure_auto_recovery(
        "unity_runtime",
        AutoRecoveryPolicy(
            enabled=settings.ops_auto_recover_unity,
            cooldown_seconds=settings.ops_auto_recover_cooldown_seconds,
            max_attempts_per_window=settings.ops_auto_recover_max_attempts_per_window,
            window_seconds=settings.ops_auto_recover_window_seconds,
        ),
    )
    lifecycle_supervisor.configure_auto_recovery(
        "voice_runtime",
        AutoRecoveryPolicy(
            enabled=settings.ops_auto_recover_voice,
            cooldown_seconds=settings.ops_auto_recover_cooldown_seconds,
            max_attempts_per_window=settings.ops_auto_recover_max_attempts_per_window,
            window_seconds=settings.ops_auto_recover_window_seconds,
        ),
    )
    lifecycle_supervisor.configure_auto_recovery(
        "vision_runtime",
        AutoRecoveryPolicy(
            enabled=settings.ops_auto_recover_vision,
            cooldown_seconds=settings.ops_auto_recover_cooldown_seconds,
            max_attempts_per_window=settings.ops_auto_recover_max_attempts_per_window,
            window_seconds=settings.ops_auto_recover_window_seconds,
        ),
    )
    autonomy_executor._runtime = runtime_service  # noqa: SLF001
    state_manager.register_service("jarvis.app")
    state_manager.register_service("memory")
    state_manager.register_service("modules")
    state_manager.register_service("providers")
    state_manager.register_service("models")
    state_manager.register_service("embedding_providers")
    state_manager.register_service("semantic_embeddings")
    state_manager.register_service("automation")
    state_manager.register_service("task_router")
    state_manager.register_service("runtime")
    state_manager.register_service("ops_runtime")
    state_manager.register_service("semantic_memory")
    state_manager.register_service("indexing_runtime")
    state_manager.register_service("hud_runtime")
    state_manager.register_service("ui_automation")
    state_manager.register_service("vision_runtime")
    state_manager.register_service("voice_runtime")
    state_manager.register_service("system_runtime")
    state_manager.register_service("unity_runtime")
    state_manager.register_service("autonomy")
    state_manager.register_service("research_runtime")
    state_manager.register_service("science_runtime")
    state_manager.register_service("security_runtime")
    state_manager.register_service("self_improvement_runtime")
    state_manager.register_service("writing_runtime")
    state_manager.register_service("desktop_agent_runtime")
    return JarvisApplication(
        settings=settings,
        event_bus=event_bus,
        mode_manager=mode_manager,
        state_manager=state_manager,
        capability_registry=capability_registry,
        action_registry=action_registry,
        action_router=router,
        tool_registry=tool_registry,
        provider_registry=provider_registry,
        model_catalog=model_catalog,
        model_router=model_router,
        model_service=model_service,
        embedding_provider_registry=embedding_provider_registry,
        embedding_router=embedding_router,
        embedding_service=embedding_service,
        ui_automation_service=ui_automation_service,
        vision_runtime_service=vision_runtime_service,
        voice_runtime_service=voice_runtime_service,
        system_runtime_service=system_runtime_service,
        unity_runtime_service=unity_runtime_service,
        autonomy_service_runtime=autonomy_service_runtime,
        research_runtime_service=research_runtime_service,
        science_runtime_service=science_runtime_service,
        security_runtime_service=security_runtime_service,
        self_improvement_runtime_service=self_improvement_runtime_service,
        writing_runtime_service=writing_runtime_service,
        desktop_agent_runtime_service=desktop_agent_runtime_service,
        meta_command_parser=meta_command_parser,
        task_router=task_router,
        memory_service=memory_service,
        semantic_memory_service=semantic_memory_service,
        indexing_runtime_service=indexing_runtime_service,
        hud_runtime_service=hud_runtime_service,
        mission_persistence_service=mission_persistence,
        module_registry=module_registry,
        automation_service=automation_service,
        orchestrator=orchestrator,
        runtime_service=runtime_service,
        ops_runtime_service=ops_runtime_service,
        lifecycle_supervisor=lifecycle_supervisor,
        logger=logger,
    )
