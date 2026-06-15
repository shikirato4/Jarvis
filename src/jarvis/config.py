from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.core.modes import ExecutionMode

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_DEFAULT_ENV_FILE),
        env_prefix="JARVIS_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Jarvis"
    environment: str = "local"
    log_level: str = "INFO"
    json_logs: bool = True
    data_dir: Path = Path("runtime")
    logs_dir: Path | None = None
    log_file_name: str = "jarvis.log"
    log_file_max_bytes: int = 5_000_000
    log_file_backup_count: int = 5
    workspace_root: Path = Field(default_factory=Path.cwd)
    database_url: str | None = None
    command_allowlist: tuple[str, ...] = ()
    research_allowed_roots: tuple[Path, ...] = ()
    research_default_extensions: tuple[str, ...] = (
        ".md",
        ".py",
        ".txt",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
    )
    research_max_file_size_bytes: int = 1_000_000
    automation_timezone: str = "UTC"
    state_history_limit: int = 50
    ops_event_history_limit: int = 200
    ops_snapshot_history_limit: int = 50
    ops_receipt_retention_limit: int = 100
    ops_default_timeout_ms: int = 5_000
    ops_health_timeout_ms: int = 1_500
    ops_recovery_timeout_ms: int = 10_000
    ops_shutdown_timeout_ms: int = 5_000
    ops_retry_budget_window_seconds: float = 60.0
    ops_retry_budget_max_retries: int = 10
    ops_retry_max_attempts: int = 2
    ops_retry_base_backoff_seconds: float = 0.1
    ops_retry_max_backoff_seconds: float = 1.5
    ops_breaker_failure_threshold: int = 3
    ops_breaker_recovery_timeout_seconds: float = 30.0
    ops_slow_operation_threshold_ms: float = 1_000.0
    ops_watchdog_poll_interval_seconds: float = 1.0
    ops_resource_poll_interval_seconds: float = 5.0
    ops_operation_history_limit: int = 200
    ops_default_max_concurrent_operations: int = 8
    ops_default_queue_limit: int = 16
    ops_retention_max_age_seconds: float = 86_400.0
    ops_snapshot_retention_max_age_seconds: float = 604_800.0
    ops_log_retention_max_age_seconds: float = 1_209_600.0
    ops_auto_recover_models: bool = True
    ops_auto_recover_voice: bool = True
    ops_auto_recover_vision: bool = True
    ops_auto_recover_system: bool = False
    ops_auto_recover_unity: bool = False
    ops_auto_recover_cooldown_seconds: float = 60.0
    ops_auto_recover_max_attempts_per_window: int = 2
    ops_auto_recover_window_seconds: float = 300.0
    default_mode: ExecutionMode = ExecutionMode.ASSIST
    sqlite_busy_timeout_ms: int = 5_000
    sqlite_pool_pre_ping: bool = True
    sqlite_check_same_thread: bool = False
    model_provider_default: str = "ollama"
    model_provider_fallback_order: tuple[str, ...] = ("ollama", "gpt_oss")
    general_chat_model_provider: str = "ollama"
    general_chat_model_fallback_order: tuple[str, ...] = ("gpt_oss",)
    gpt_oss_enabled: bool = False
    gpt_oss_base_url: str = "http://127.0.0.1:8000/v1"
    gpt_oss_api_key: str | None = None
    gpt_oss_timeout_seconds: float = 90.0
    gpt_oss_healthcheck_timeout_seconds: float = 5.0
    gpt_oss_chat_endpoint: str = "/chat/completions"
    gpt_oss_models_endpoint: str = "/models"
    gpt_oss_general_model: str = "gpt-oss-20b"
    gpt_oss_reasoning_model: str = "gpt-oss-20b"
    gpt_oss_coding_model: str = "gpt-oss-20b"
    gpt_oss_summarizer_model: str = "gpt-oss-20b"
    gpt_oss_writing_model: str = "gpt-oss-20b"
    gpt_oss_planner_model: str = "gpt-oss-20b"
    embedding_provider_default: str = "ollama_embeddings"
    embedding_provider_fallback_order: tuple[str, ...] = ("ollama_embeddings",)
    embedding_model_default: str = "nomic-embed-text"
    embeddings_enabled: bool = True
    ollama_embeddings_endpoint: str = "/api/embed"
    semantic_chunk_size: int = 900
    semantic_chunk_overlap: int = 120
    semantic_top_k: int = 5
    semantic_min_score: float = 0.2
    semantic_reranking_type: str = "basic"
    semantic_context_char_budget: int = 4_000
    semantic_max_chunks_per_document: int = 200
    semantic_context_max_chunks: int = 8
    semantic_degradation_policy: str = "embedding_fallback_to_lexical_then_operational"
    ui_backend_kind: str = "windows"
    ui_blocked_window_titles: tuple[str, ...] = ("jarvis",)
    ui_allowed_window_titles: tuple[str, ...] = (
        "word",
        "excel",
        "powerpoint",
        "outlook",
        "notepad",
        "vscode",
        "visual studio code",
        "chrome",
        "google chrome",
        "opera",
        "opera gx",
        "calculadora",
        "calculator",
        "explorador de archivos",
        "explorer",
        "file explorer",
    )
    ui_require_confirmation_for_unknown_windows: bool = True
    ui_allow_discovered_applications: bool = True
    ui_hotkey_blocklist: tuple[str, ...] = ("alt+f4", "win+l")
    ui_default_block_size: int = 280
    ui_default_typing_interval_ms: int = 8
    ui_default_pause_between_blocks_ms: int = 120
    ui_max_text_length: int = 20_000
    ui_max_blocks_per_operation: int = 80
    ui_max_actions_per_minute: int = 120
    ui_watchdog_timeout_ms: int = 15_000
    ui_focus_timeout_seconds: float = 3.0
    ui_mouse_move_default_duration_seconds: float = 0.15
    ui_direct_write_requires_operator_mode: bool = True
    ui_direct_write_max_block_size: int = 500
    ui_copilot_pause_between_blocks_ms: int = 200
    writing_operation_timeout_ms: int = 180_000
    writing_context_timeout_ms: int = 15_000
    writing_generation_timeout_ms: int = 90_000
    writing_ui_write_timeout_ms: int = 90_000
    writing_ui_block_size: int = 500
    writing_ui_typing_interval_ms: int = 2
    writing_ui_pause_between_blocks_ms: int = 40
    voice_audio_input_backend_default: str = "sounddevice"
    voice_audio_output_backend_default: str = "winsound"
    voice_stt_provider_default: str = "in_memory"
    voice_stt_provider_fallback_order: tuple[str, ...] = ("in_memory",)
    voice_tts_provider_default: str = "coqui_xtts"
    voice_tts_provider_fallback_order: tuple[str, ...] = ("pyttsx3", "in_memory")
    voice_default_language: str = "es"
    voice_coqui_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voice_coqui_device_preference: str = "auto"
    voice_coqui_speaker_wav: Path | None = None
    voice_coqui_speaker_name: str | None = None
    voice_coqui_tos_agreed: bool = False
    voice_clone_enabled: bool = True
    voice_clone_backend_default: str = ""
    voice_clone_profile_default: str = "jarvis_premium"
    voice_clone_sample_path: Path | None = None
    voice_clone_preprocess_enabled: bool = True
    voice_clone_quality_check_enabled: bool = True
    voice_clone_quality_threshold: float = 0.35
    voice_clone_openvoice_enabled: bool = False
    voice_clone_rvc_enabled: bool = False
    voice_profile_default: str = "jarvis_premium"
    voice_style_preset: str = "cinematic"
    voice_pause_style: str = "balanced"
    voice_speaking_rate: float = 0.90
    voice_formality_level: int = 5
    voice_tts_rate: int = 168
    voice_cleanup_enabled: bool = True
    voice_style_enabled: bool = True
    voice_enabled: bool = True
    voice_start_muted: bool = False
    voice_input_enabled: bool = True
    voice_input_provider_default: str = "faster_whisper"
    voice_input_language: str = "es"
    voice_input_timeout_seconds: float = 4.0
    voice_input_silence_threshold: float = 0.02
    voice_push_to_talk_default: bool = False
    voice_wakeword_enabled: bool = False
    voice_input_start_muted: bool = False
    voice_clap_sensitivity: float = 0.7
    voice_clap_cooldown_seconds: float = 0.15
    voice_clap_window_seconds: float = 0.8
    voice_silence_threshold: float = 0.02
    voice_speech_threshold: float = 0.05
    voice_buffer_chunk_seconds: float = 1.0
    voice_default_listen_seconds: float = 3.0
    voice_default_session_mode: str = "listen"
    voice_degradation_policy: str = "fallback_stt_tts_then_text"
    voice_max_session_seconds: float = 30.0
    voice_watchdog_timeout_ms: int = 45_000
    voice_default_rate: int = 180
    voice_default_voice_name: str | None = None
    voice_cancel_phrases: tuple[str, ...] = ("alto", "detente", "cancela", "stop", "cancel")
    voice_command_auto_route: bool = False
    voice_listen_playback_response: bool = False
    vision_enabled: bool = True
    vision_capture_backend_default: str = "windows_mss"
    vision_ocr_provider_default: str = "in_memory_ocr"
    vision_ocr_provider_fallback_order: tuple[str, ...] = ("in_memory_ocr",)
    vision_awareness_backend_default: str = "heuristic_vision"
    vision_max_image_width: int = 3840
    vision_max_image_height: int = 2160
    vision_max_capture_area: int = 8_294_400
    vision_max_image_bytes: int = 12_000_000
    vision_ocr_max_blocks: int = 500
    vision_watchdog_timeout_ms: int = 20_000
    vision_default_ocr_language: str = "spa"
    vision_store_captures: bool = False
    vision_capture_dir: Path | None = None
    vision_redact_sensitive_regions: bool = False
    vision_degradation_policy: str = "awareness_fallback_to_ocr_then_ui_tree_then_metadata"
    autonomy_enabled: bool = True
    autonomy_default_level: str = "assisted"
    autonomy_max_steps: int = 12
    autonomy_max_duration_seconds: float = 180.0
    autonomy_max_concurrent_missions: int = 4
    autonomy_watchdog_timeout_ms: int = 60_000
    autonomy_max_replans: int = 3
    autonomy_max_retries_per_step: int = 2
    autonomy_default_verification_policy: str = "receipt_plus_observation"
    autonomy_high_risk_requires_confirmation: bool = True
    autonomy_default_strategy: str = "balanced"
    autonomy_stop_on_low_confidence: bool = True
    autonomy_reflection_enabled: bool = True
    system_runtime_enabled: bool = True
    system_backend_kind: str = "native"
    system_search_roots: tuple[Path, ...] = ()
    system_known_locations: dict[str, str] = Field(default_factory=dict)
    system_search_max_depth: int = 5
    system_search_max_results: int = 20
    system_search_max_nodes: int = 5_000
    system_operation_timeout_ms: int = 10_000
    system_max_path_length: int = 4096
    system_search_excluded_dirnames: tuple[str, ...] = (
        "$recycle.bin",
        "system volume information",
        "__pycache__",
        ".git",
        "node_modules",
    )
    system_blocked_extensions: tuple[str, ...] = (".bat", ".cmd", ".ps1", ".vbs", ".js", ".jse", ".wsf", ".wsh", ".reg")
    system_allowed_uri_schemes: tuple[str, ...] = ("http", "https", "mailto", "file")
    system_blocked_uri_schemes: tuple[str, ...] = ("javascript", "data", "vbscript", "ms-settings")
    system_require_confirmation_for_launch: bool = True
    system_require_confirmation_for_system_roots: bool = True
    system_require_confirmation_for_executable_open: bool = True
    system_require_confirmation_for_network_or_removable: bool = True
    system_sensitive_roots: tuple[Path, ...] = ()
    system_allowed_application_ids: tuple[str, ...] = ()
    system_allowed_executables: tuple[str, ...] = ()
    system_blocked_paths: tuple[Path, ...] = ()
    unity_runtime_enabled: bool = True
    unity_discovery_roots: tuple[Path, ...] = ()
    unity_known_locations: dict[str, str] = Field(default_factory=dict)
    unity_search_max_depth: int = 6
    unity_search_max_results: int = 50
    unity_project_allowed_roots: tuple[Path, ...] = ()
    unity_project_blocked_roots: tuple[Path, ...] = ()
    unity_allowed_installation_paths: tuple[Path, ...] = ()
    unity_require_confirmation_for_project_creation: bool = True
    unity_require_confirmation_for_script_overwrite: bool = True
    unity_require_confirmation_for_editor_open: bool = True
    unity_require_confirmation_for_bridge_commands: bool = True
    unity_blocked_asset_paths: tuple[Path, ...] = ()
    unity_blocked_extensions: tuple[str, ...] = (".dll", ".exe", ".bat", ".cmd", ".ps1")
    unity_default_scripts_folder: str = "Assets/Scripts"
    unity_default_namespace: str | None = None
    unity_bridge_enabled: bool = True
    unity_bridge_backend_default: str = "noop"
    unity_bridge_transport_default: str = "stub"
    unity_bridge_host: str = "127.0.0.1"
    unity_bridge_port: int = 57991
    unity_bridge_timeout_ms: int = 2_500
    unity_bridge_retry_count: int = 1
    unity_bridge_watchdog_timeout_ms: int = 10_000
    unity_max_sessions: int = 8
    unity_launch_strategy_default: str = "direct_editor"
    unity_hub_path: str | None = None
    unity_editor_fallback_paths: tuple[str, ...] = ()
    unity_auto_connect_bridge: bool = True
    unity_require_confirmation_for_launch: bool = True
    semantic_limits_by_source_type: dict[str, int] = Field(
        default_factory=lambda: {
            "book": 500,
            "pdf": 400,
            "markdown": 200,
            "research_note": 150,
            "draft": 150,
            "note": 100,
            "text": 150,
            "json": 150,
        }
    )
    ollama_enabled: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 90.0
    ollama_healthcheck_timeout_seconds: float = 5.0
    chat_default_max_tokens: int = 500
    chat_fast_max_tokens: int = 160
    chat_detailed_max_tokens: int = 1200
    chat_temperature: float = 0.4
    llm_timeout_seconds: float = 60.0
    llm_fast_timeout_seconds: float = 20.0
    llm_detailed_timeout_seconds: float = 120.0
    web_synthesis_timeout_seconds: float = 75.0
    web_synthesis_max_sources: int = 3
    web_synthesis_snippet_chars: int = 500
    research_visible_sources: int = 5
    research_synthesis_max_tokens: int = 900
    research_short_max_tokens: int = 220
    ollama_keep_warm: bool = False
    ollama_keep_warm_prompt: str = "ping"
    ollama_keep_warm_delay_seconds: float = 5.0
    ollama_max_retries: int = 2
    ollama_retry_backoff_seconds: float = 0.25
    ollama_chat_endpoint: str = "/api/chat"
    ollama_tags_endpoint: str = "/api/tags"
    ollama_stream_endpoint: str = "auto"
    unity_allowed_bridge_commands: tuple[str, ...] = ()
    unity_blocked_bridge_commands: tuple[str, ...] = ()
    unity_require_confirmation_for_editor_commands: bool = True
    unity_require_confirmation_for_custom_commands: bool = True
    unity_allow_stub_when_bridge_unavailable: bool = True
    unity_launch_arguments: tuple[str, ...] = ()
    research_runtime_enabled: bool = True
    research_default_collection: str = "deep_research"
    research_default_max_steps: int = 12
    research_default_max_duration_seconds: float = 90.0
    research_default_max_sources: int = 6
    research_default_max_findings: int = 20
    research_max_concurrent_tasks: int = 3
    research_watchdog_timeout_ms: int = 45_000
    research_require_citations: bool = True
    indexing_runtime_enabled: bool = True
    indexing_auto_sync_on_start: bool = False
    indexing_auto_index_research: bool = True
    indexing_auto_index_writing: bool = True
    indexing_default_batch_size: int = 32
    indexing_max_concurrent_runs: int = 2
    indexing_default_chunk_size: int = 900
    indexing_default_chunk_overlap: int = 120
    indexing_max_chunks_per_document: int = 200
    indexing_max_file_size_bytes: int = 2_000_000
    indexing_snapshot_retention: int = 20
    indexing_degradation_policy: str = "incremental_sync_with_semantic_projection"
    indexing_allowed_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
        ".py",
        ".cs",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".cpp",
        ".h",
        ".hpp",
        ".unity",
        ".prefab",
        ".asset",
        ".asmdef",
        ".pdf",
    )
    indexing_sensitive_name_patterns: tuple[str, ...] = (
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*.pfx",
        "*.kdbx",
        "*secret*",
        "*credential*",
        "*token*",
    )
    indexing_excluded_dirnames: tuple[str, ...] = (
        ".git",
        ".venv",
        ".venv312",
        ".venv_legacy_src_backup",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "Library",
        "Temp",
        "Logs",
        "obj",
        "bin",
    )
    hud_enabled: bool = True
    hud_poll_interval_ms: int = 5000

    @field_validator("data_dir", "logs_dir", "workspace_root", "vision_capture_dir", "voice_coqui_speaker_wav", "voice_clone_sample_path", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path | None) -> Path | None:
        if value is None:
            return value
        return Path(value).expanduser()

    @field_validator(
        "research_allowed_roots",
        "system_search_roots",
        "system_sensitive_roots",
        "system_blocked_paths",
        "unity_discovery_roots",
        "unity_project_allowed_roots",
        "unity_project_blocked_roots",
        "unity_allowed_installation_paths",
        "unity_blocked_asset_paths",
        mode="before",
    )
    @classmethod
    def _coerce_roots(cls, value: object) -> tuple[Path, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            parts = [item.strip() for item in value.split(",") if item.strip()]
            return tuple(Path(item).expanduser() for item in parts)
        if isinstance(value, (list, tuple)):
            return tuple(Path(item).expanduser() for item in value)
        raise TypeError("research_allowed_roots must be a string or a sequence of paths")

    @field_validator("command_allowlist", mode="before")
    @classmethod
    def _coerce_allowlist(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("command_allowlist must be a string or a sequence of strings")

    @field_validator("model_provider_fallback_order", mode="before")
    @classmethod
    def _coerce_provider_order(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("model_provider_fallback_order must be a string or a sequence of strings")

    @field_validator("general_chat_model_fallback_order", mode="before")
    @classmethod
    def _coerce_general_chat_provider_order(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("general_chat_model_fallback_order must be a string or a sequence of strings")

    @field_validator("ui_blocked_window_titles", "ui_allowed_window_titles", "ui_hotkey_blocklist", mode="before")
    @classmethod
    def _coerce_string_tuple(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("value must be a string or a sequence of strings")

    @field_validator(
        "voice_stt_provider_fallback_order",
        "voice_tts_provider_fallback_order",
        "voice_cancel_phrases",
        "vision_ocr_provider_fallback_order",
        "system_search_excluded_dirnames",
        "system_blocked_extensions",
        "system_allowed_uri_schemes",
        "system_blocked_uri_schemes",
        "system_allowed_application_ids",
        "system_allowed_executables",
        "unity_blocked_extensions",
        "unity_allowed_bridge_commands",
        "unity_blocked_bridge_commands",
        "unity_editor_fallback_paths",
        "unity_launch_arguments",
        "indexing_allowed_extensions",
        "indexing_sensitive_name_patterns",
        "indexing_excluded_dirnames",
        mode="before",
    )
    @classmethod
    def _coerce_voice_string_tuple(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("value must be a string or a sequence of strings")

    @field_validator("system_known_locations", mode="before")
    @classmethod
    def _coerce_system_known_locations(cls, value: object) -> dict[str, str]:
        if value in (None, "", ()):
            return {}
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        raise TypeError("system_known_locations must be a dictionary of location ids to paths")

    @field_validator("unity_known_locations", mode="before")
    @classmethod
    def _coerce_unity_known_locations(cls, value: object) -> dict[str, str]:
        if value in (None, "", ()):
            return {}
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        raise TypeError("unity_known_locations must be a dictionary of location ids to paths")

    @field_validator("embedding_provider_fallback_order", mode="before")
    @classmethod
    def _coerce_embedding_provider_order(cls, value: object) -> tuple[str, ...]:
        if value in (None, "", ()):
            return ()
        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        raise TypeError("embedding_provider_fallback_order must be a string or a sequence of strings")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.resolve()

    @property
    def resolved_workspace_root(self) -> Path:
        return self.workspace_root.resolve()

    @property
    def resolved_logs_dir(self) -> Path:
        if self.logs_dir is not None:
            return self.logs_dir.resolve()
        return (self.resolved_data_dir / "logs").resolve()

    @property
    def resolved_vision_capture_dir(self) -> Path:
        if self.vision_capture_dir is not None:
            return self.vision_capture_dir.resolve()
        return (self.resolved_data_dir / "captures").resolve()

    @property
    def resolved_voice_coqui_speaker_wav(self) -> Path | None:
        if self.voice_coqui_speaker_wav is None:
            return None
        return self.voice_coqui_speaker_wav.resolve()

    @property
    def resolved_voice_clone_sample_path(self) -> Path | None:
        if self.voice_clone_sample_path is None:
            return None
        return self.voice_clone_sample_path.resolve()

    @property
    def resolved_log_file(self) -> Path:
        return self.resolved_logs_dir / self.log_file_name

    @property
    def resolved_research_roots(self) -> tuple[Path, ...]:
        if self.research_allowed_roots:
            return tuple(path.resolve() for path in self.research_allowed_roots)
        return (self.resolved_workspace_root,)

    @property
    def resolved_system_search_roots(self) -> tuple[Path, ...]:
        roots = [self.resolved_workspace_root]
        roots.extend(path.resolve() for path in self.system_search_roots)
        return tuple(dict.fromkeys(roots))

    @property
    def resolved_system_sensitive_roots(self) -> tuple[Path, ...]:
        roots = [path.resolve() for path in self.system_sensitive_roots]
        return tuple(dict.fromkeys(roots))

    @property
    def resolved_system_blocked_paths(self) -> tuple[Path, ...]:
        roots = [path.resolve() for path in self.system_blocked_paths]
        return tuple(dict.fromkeys(roots))

    @property
    def resolved_unity_discovery_roots(self) -> tuple[Path, ...]:
        roots = [self.resolved_workspace_root]
        roots.extend(path.resolve() for path in self.unity_discovery_roots)
        return tuple(dict.fromkeys(roots))

    @property
    def resolved_unity_project_allowed_roots(self) -> tuple[Path, ...]:
        roots = [path.resolve() for path in self.unity_project_allowed_roots]
        if not roots:
            roots = list(self.resolved_unity_discovery_roots)
        return tuple(dict.fromkeys(roots))

    @property
    def resolved_unity_project_blocked_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(path.resolve() for path in self.unity_project_blocked_roots))

    @property
    def resolved_unity_allowed_installation_paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(path.resolve() for path in self.unity_allowed_installation_paths))

    @property
    def resolved_unity_blocked_asset_paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(path.resolve() for path in self.unity_blocked_asset_paths))

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        database_path = (self.resolved_data_dir / "jarvis.db").resolve()
        return f"sqlite+pysqlite:///{database_path.as_posix()}"

    def sqlalchemy_engine_options(self) -> dict[str, Any]:
        database_url = self.sqlalchemy_database_url
        if database_url.startswith("sqlite"):
            return {
                "pool_pre_ping": self.sqlite_pool_pre_ping,
                "connect_args": {
                    "check_same_thread": self.sqlite_check_same_thread,
                    "timeout": max(self.sqlite_busy_timeout_ms / 1000, 0.1),
                },
            }
        return {"pool_pre_ping": True}

    def prepare_environment(self) -> None:
        self.resolved_data_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_logs_dir.mkdir(parents=True, exist_ok=True)
