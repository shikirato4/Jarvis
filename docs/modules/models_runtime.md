# Models Runtime

Responsibility: expose model providers through a stable internal service with catalog selection, fallback, and observability.

Key files:
- `src/jarvis/models_runtime/base.py`
- `src/jarvis/models_runtime/catalog.py`
- `src/jarvis/models_runtime/registry.py`
- `src/jarvis/models_runtime/router.py`
- `src/jarvis/models_runtime/service.py`
- `src/jarvis/models_runtime/ollama.py`

Failure points:
- provider unavailability
- invalid fallback chains
- timeouts and retry storms
- parsing failures on structured model output
- catalog drift between logical names and real provider models
