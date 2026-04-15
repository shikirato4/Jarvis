# Extending Jarvis

Use these extension seams:

- Add shared request or response contracts in `src/jarvis/models/`.
- Add new runtime policy or state rules in `src/jarvis/core/`.
- Add reusable lower-level callable capability in `src/jarvis/tools/`.
- Add a business-level operation in `src/jarvis/actions/`.
- Add provider implementations and routing rules in `src/jarvis/models_runtime/`.
- Add a concrete capability provider in `src/jarvis/modules/`.
- Register the module in `src/jarvis/bootstrap.py`.

Prefer action-plus-tool registration when a capability needs both structured workflow execution and direct low-level invocation.
Prefer `ModelService` injection when a feature needs inference; do not couple modules directly to Ollama.
