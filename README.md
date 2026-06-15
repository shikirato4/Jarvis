# Jarvis Cognitive OS

Jarvis is the Phase 1 core of a local cognitive operating system. The codebase is organized as a runtime kernel, not as a toy chatbot, and is ready to grow into a desktop application, a local daemon, or an embedded service.

## Source Tree

```text
src/jarvis/
  __main__.py
  bootstrap.py
  cli.py
  config.py
  logging.py
  actions/
    base.py
    models.py
    registry.py
    router.py
  api/
    app.py
  automation/
    service.py
  cognition/
    models.py
    orchestrator.py
  core/
    capabilities.py
    errors.py
    events.py
    metacommands.py
    models.py
    modes.py
    modules.py
    process.py
    safety.py
    services.py
    state.py
  memory/
    base.py
    models.py
    repository.py
    service.py
  models/
    base.py
  models_runtime/
    base.py
    catalog.py
    ollama.py
    registry.py
    router.py
    service.py
  modules/
    memory_module.py
    operations_module.py
    research_module.py
    vision_module.py
    voice_module.py
    writer_module.py
  routing/
    models.py
    task_router.py
  services/
    runtime.py
  tools/
    builtin.py
    models.py
    registry.py
```

## Core Dependencies

- `pydantic` and `pydantic-settings`: shared contracts, typed payload validation, and centralized configuration.
- `SQLAlchemy`: local persistence for memories, activity records, and automation state.
- `APScheduler`: recurring execution for background automations.
- `FastAPI`: local HTTP service surface for a future desktop shell or local agents.
- `Typer`: operational CLI on top of the same runtime service facade.
- `Pillow`: base image inspection capability for the vision layer.

## Runtime Layers

- `config.py`: central settings, environment preparation, paths, logging, and safety limits.
- `core/`: errors, events, runtime state, mode policy, metacommand parsing, and service contracts.
- `models/`: common strict Pydantic base models.
- `actions/`: executable action contracts, validation, receipts, and rollback-aware routing.
- `tools/`: lower-level tool calling layer for direct capability access.
- `models_runtime/`: provider abstraction, model catalog, model routing and Ollama integration.
- `routing/`: top-level task router that decides between metacommands, tools, actions, and orchestration.
- `cognition/`: orchestration kernel for intent-to-plan resolution.
- `memory/`: persistence backbone for memories, activity, and automations.
- `modules/`: concrete capability providers.
- `services/`: stable runtime facade for CLI, API, and future desktop UI.

## General Execution Flow

1. `build_application()` in `src/jarvis/bootstrap.py` creates the full dependency graph.
2. `Settings` prepares runtime and log directories.
3. Logging is configured with structured console output plus a rotating file handler.
4. Registries are created for actions, tools, modules, capabilities, providers, modes, runtime state, and services.
5. Capability modules register actions and capability descriptors, while builtin tools register tool-call adapters.
6. `ModelService` sits behind `ProviderRegistry` and `ModelRouter`, so neither tools, actions, nor orchestration call Ollama directly.
7. `JarvisApplication.start()` creates the persistence schema, starts modules, starts automation, and brings the runtime service online.
8. The top-level request path goes through `TaskRouter`, which can:
   - parse a metacommand such as `/mode operator`
   - invoke a registered tool
   - execute a direct action
   - hand a freeform request to the cognitive orchestrator
9. The cognitive orchestrator can classify intent, generate plans, and summarize findings through `ModelService`, with controlled fallback to descriptor heuristics if providers are unavailable.
10. The state manager records services, tasks, tool invocations, model invocations, and mode changes for inspection.

## Main Entry Points

- `src/jarvis/bootstrap.py`: system bootstrap and runtime assembly.
- `src/jarvis/cli.py`: command-line entry point.
- `src/jarvis/api/app.py`: HTTP entry point.
- `src/jarvis/__main__.py`: module execution entry point.

## Iniciar la Interfaz Desktop

JARVIS incluye una interfaz de control desktop oficial. Inicia la aplicación de forma segura con:

```powershell
python -m jarvis.desktop
```

El arranque es autoconfigurable: detecta tu conexión a internet y tu proveedor local de modelos (por defecto Ollama en `http://127.0.0.1:11434`), y ajusta la interfaz automáticamente en modo `auto`, `offline` o `disabled`.

Si deseas diagnosticar el estado del entorno de red y los modelos detectados, usa el comando de diagnóstico:

```powershell
python -m jarvis doctor
```
o su equivalente para el Code Agent:
```powershell
python -m jarvis code doctor
```

## Local Image Runtime

Jarvis can generate images locally with JuggernautXL SDXL through Diffusers. Fooocus is not used as the runtime; the Fooocus checkpoint can be copied into the Jarvis workspace and loaded directly by Diffusers.

Expected local checkpoint path:

```text
models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors
```

Large model files are ignored by Git. Do not commit `.safetensors`, `.ckpt`, `.bin`, `.pt`, `.pth`, generated images, `.env`, API keys, or output folders.

Useful commands:

```powershell
python -m jarvis image status
python -m jarvis image generate --prompt "a futuristic blue Jarvis orb made of glowing particles"
python -m jarvis image cancel
python -m jarvis image unload
```

Optional image dependencies:

```powershell
python -m pip install -e .[image]
```

The first generation can take a long time because the SDXL checkpoint is loaded lazily. The model remains cached when possible and can be unloaded with `python -m jarvis image unload`.

## Example Commands

```bash
python -m pip install -e .[dev]
python -m jarvis describe
python -m jarvis state
python -m jarvis models
python -m jarvis providers
python -m jarvis infer "Classify this request"
python -m jarvis task --input "/mode operator"
python -m jarvis tool memory.lookup --payload "{\"query\": \"jarvis\", \"limit\": 5}"
python -m jarvis tool model.chat --payload "{logical_model: 'general_assistant', prompt: 'Summarize the workspace state'}"
python -m jarvis serve
```

## Python Environment

Keep the virtual environment at the repository root, not inside `src/jarvis`.

Recommended setup on Windows:

```powershell
cd C:\Users\GAMER\Documents\jarvis
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[desktop,voice]
python -m pip install -e .[dev]
```

Recommended validation commands:

```powershell
python --version
python -c "import jarvis; print('ok')"
python -m compileall src\jarvis
python -m pytest tests/test_voice_runtime.py -q
python -m pytest tests/test_desktop_runtime_routing.py -q
```

Remove any legacy repo-local virtual environment backups after the root `.venv` is confirmed working. Keeping virtual environments or archived copies inside the indexed workspace will slow down searches, validation, and indexing.

## Registering New Capabilities

New capabilities should enter the runtime as modules, actions, and optionally tools. The recommended pattern is:

1. Create a new module under `src/jarvis/modules/`.
2. Register one or more actions in `register_actions()`.
3. Optionally register tools in `register_tools()`.
4. Add the module to `build_application()`.
5. If the capability needs reasoning, use `context.models` or inject `ModelService`; do not call Ollama directly.

Example:

```python
from pydantic import BaseModel

from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.tools.models import ToolResult
from jarvis.tools.registry import ToolContext, ToolDefinition, ToolRegistry


class DiagnosticsPayload(BaseModel):
    detail: str = "basic"


class DiagnosticsModule:
    name = "diagnostics"
    description = "Runtime diagnostics and health analysis."

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(
            ActionDefinition(
                name="diagnostics.health_report",
                description="Build a runtime health report.",
                payload_model=DiagnosticsPayload,
                handler=self._health_report,
                tags=("diagnostics", "health"),
            )
        )

    def register_tools(self, registry: ToolRegistry) -> None:
        registry.register(
            ToolDefinition(
                name="diagnostics.health",
                description="Direct health inspection tool.",
                input_model=DiagnosticsPayload,
                handler=self._health_tool,
                tags=("diagnostics", "state"),
            )
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def _health_report(self, context: ActionContext, payload: DiagnosticsPayload) -> ActionResult:
        return ActionResult(message="health report ready", data={"detail": payload.detail})

    def _health_tool(self, context: ToolContext, payload: DiagnosticsPayload) -> ToolResult:
        return ToolResult(message="health tool ready", data={"detail": payload.detail})
```

## Current Phase Scope

The current codebase now includes Phase 2 inference wiring with Ollama behind a provider abstraction. No final visual desktop shell is included yet. The next safe expansions are STT/TTS providers, semantic vision, embeddings, reranking, memory indexing, and a desktop surface on top of the runtime service facade.
