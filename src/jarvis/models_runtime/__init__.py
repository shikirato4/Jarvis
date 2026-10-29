"""Model providers, registry, routing and service layer."""

from .catalog import ModelCatalog, ModelProfile, build_default_model_catalog
from .adapters import (
    LocalOpenAIAdapter,
    OllamaAdapter,
    LlamaCppAdapter,
    VLLMAdapter,
    OpenAICompatibleAdapter,
)
from .coordinator import HybridProviderUnavailableError, PersistentModeStore, ProviderCoordinator
from .hybrid import (
    LLMMode,
    ProviderCapabilities,
    ProviderOperationalState,
    ProviderSelection,
    ProviderStatus,
    RequestExecutionContext,
)
from .nemotron_local import NemotronOmniLocalProvider
from .model_identity import ConfiguredModelIdentity
from .qwen_local import QwenLocalProvider
from .local_model_runtime import (
    LocalModelRuntimeManager,
    LocalRuntimeBackend,
    LocalRuntimeOwnership,
    LocalRuntimeState,
    LocalRuntimeStatus,
)
from .local_profiles import (
    BASELINE_PROFILE_ID,
    CONFIGURED_PROFILE_ID,
    QWEN35_PROFILE_ID,
    LocalModelProfile,
    LocalModelProfileConfigurationError,
    baseline_local_model_profile,
    configured_local_model_profile,
    resolve_local_model_profile,
    registered_local_model_profiles,
)
from .nvidia_nim import NvidiaNimProvider
from .registry import ProviderRegistry
from .router import ModelRouter
from .service import ModelService
from .roles import CapabilityState, LLMTaskRouter, ModelRoleRegistry, build_model_role_registry

__all__ = [
    "LocalOpenAIAdapter",
    "OllamaAdapter",
    "LlamaCppAdapter",
    "VLLMAdapter",
    "OpenAICompatibleAdapter",
    "ModelCatalog",
    "CapabilityState",
    "LLMTaskRouter",
    "ModelRoleRegistry",
    "build_model_role_registry",
    "ModelProfile",
    "ModelRouter",
    "ModelService",
    "ProviderRegistry",
    "LLMMode",
    "ProviderCapabilities",
    "ProviderOperationalState",
    "ProviderSelection",
    "ProviderStatus",
    "RequestExecutionContext",
    "NemotronOmniLocalProvider",
    "ConfiguredModelIdentity",
    "QwenLocalProvider",
    "LocalModelRuntimeManager",
    "LocalRuntimeBackend",
    "LocalRuntimeOwnership",
    "LocalRuntimeState",
    "LocalRuntimeStatus",
    "LocalModelProfile",
    "LocalModelProfileConfigurationError",
    "BASELINE_PROFILE_ID",
    "CONFIGURED_PROFILE_ID",
    "QWEN35_PROFILE_ID",
    "baseline_local_model_profile",
    "configured_local_model_profile",
    "resolve_local_model_profile",
    "registered_local_model_profiles",
    "NvidiaNimProvider",
    "build_default_model_catalog",
    "HybridProviderUnavailableError",
    "PersistentModeStore",
    "ProviderCoordinator",
]
