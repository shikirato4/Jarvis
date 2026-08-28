from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import Field

from jarvis.models.base import JarvisBaseModel


class LLMMode(StrEnum):
    AUTO = "auto"
    NVIDIA_API = "nvidia_api"
    LOCAL = "local"


class ProviderOperationalState(StrEnum):
    CLOUD_HEALTHY = "cloud_healthy"
    CLOUD_UNAVAILABLE = "cloud_unavailable"
    CLOUD_RATE_LIMITED = "cloud_rate_limited"
    CLOUD_INVALID_KEY = "cloud_invalid_key"
    LOCAL_ACTIVE = "local_active"
    LOCAL_UNAVAILABLE = "local_unavailable"
    BOTH_UNAVAILABLE = "both_unavailable"
    RECOVERING = "recovering"


class ProviderCapabilities(JarvisBaseModel):
    text: bool = True
    images: bool = False
    audio: bool = False
    video: bool = False
    documents: bool = False
    streaming: bool = False
    reasoning: bool = False
    structured_output: bool = False
    tool_calling: bool = False
    max_context_tokens: int | None = None
    max_files: int | None = None
    input_formats: tuple[str, ...] = ()
    native_transport_modalities: tuple[str, ...] = ()
    output_formats: tuple[str, ...] = ("text/plain",)
    verification_source: str | None = None

    def supports_modality(self, modality: str) -> bool:
        normalized = modality.strip().casefold().rstrip("s")
        fields = {
            "text": self.text,
            "image": self.images,
            "audio": self.audio,
            "video": self.video,
            "document": self.documents,
        }
        return fields.get(normalized, False)

    def supports_native_transport(self, modality: str) -> bool:
        normalized = modality.strip().casefold().rstrip("s")
        return normalized in {
            item.strip().casefold().rstrip("s")
            for item in self.native_transport_modalities
        }


class ProviderStatus(JarvisBaseModel):
    provider_name: str
    state: ProviderOperationalState
    healthy: bool
    latency_ms: float | None = None
    message: str | None = None
    retry_after_seconds: float | None = None
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProviderSelection(JarvisBaseModel):
    selected_mode: LLMMode
    active_provider: str | None = None
    request_provider: str | None = None
    active_model: str | None = None
    provider_status: ProviderStatus
    capabilities: ProviderCapabilities = Field(default_factory=ProviderCapabilities)
    fallback_reason: str | None = None
    persistence_warning: str | None = None


class RequestExecutionContext(JarvisBaseModel):
    turn_id: str = Field(default_factory=lambda: f"turn-{uuid4()}")
    request_id: str = Field(default_factory=lambda: f"request-{uuid4()}")
    registered_file_ids: set[str] = Field(default_factory=set)
    registered_tool_call_ids: set[str] = Field(default_factory=set)
    external_effect_ids: set[str] = Field(default_factory=set)

    def register_file(self, file_id: str) -> bool:
        if file_id in self.registered_file_ids:
            return False
        self.registered_file_ids.add(file_id)
        return True

    def register_tool_call(
        self, tool_call_id: str, *, external_effect: bool = False
    ) -> bool:
        if tool_call_id in self.registered_tool_call_ids:
            return False
        self.registered_tool_call_ids.add(tool_call_id)
        if external_effect:
            self.external_effect_ids.add(tool_call_id)
        return True

    @property
    def has_external_effects(self) -> bool:
        return bool(self.external_effect_ids)

    def can_retry_stream(self, *, emitted_content: bool) -> bool:
        return not emitted_content and not self.has_external_effects
