"""Model providers, registry, routing and service layer."""

from .catalog import ModelCatalog, ModelProfile, build_default_model_catalog
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
from .nvidia_nim import NvidiaNimProvider
from .registry import ProviderRegistry
from .router import ModelRouter
from .service import ModelService

__all__ = [
    "ModelCatalog",
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
    "NvidiaNimProvider",
    "build_default_model_catalog",
    "HybridProviderUnavailableError",
    "PersistentModeStore",
    "ProviderCoordinator",
]
