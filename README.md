# Jarvis Cognitive OS

Jarvis is a local cognitive operating system organized around a shared runtime kernel. The same services back its desktop interface, CLI, HTTP API, automation, memory, model routing, voice, vision, system, research, writing, science, security, indexing, autonomy, and code-agent capabilities — plus an integrated real browser (Playwright), realtime voice conversation (Gemini Live), a face-guard that locks the workstation when a banned face appears, and local image generation with Qwen-Image-2.1-Turbo (legible text/infographics) or JuggernautXL SDXL.

## Cambios recientes

### 2026-10-25 — Navegador real, voz en vivo, guardián de caras y Qwen-Image

- **Navegador integrado (nivel 3)**: Playwright con sesión persistente (los logins sobreviven arranques), referencias de accesibilidad `[N]` para que el agente cite elementos exactos, captura de descargas con ruta de retorno, `browser.back`, `browser.find` y multi-pestaña. Acciones con alias en español (`abre youtube`, `regresa`, `busca en la página`).
- **Agente consciente de sus herramientas**: guías por familia de acción en español (`_FAMILY_GUIDE_ES`), `semantic_aliases` en todas las acciones y arranque de llama-server con `--jinja` para que las herramientas funcionen de verdad con el modelo local. Verificado E2E: el modelo local abrió YouTube solo.
- **Conversación en tiempo real por voz ("Jarvis, modo live")**: Gemini Live vía WebSocket `v1beta` (`BidiGenerateContent`), audio subido como `realtimeInput.audio`, cierre de turno por VAD del servidor. Launcher en `scripts/jarvis_live.py`. Frases como "modo live" o "conversación natural" enrutan al agente y activan el modo.
- **Guardián de caras prohibidas** (`face_guard/`): "Jarvis, banea esa cara" → la webcam toma fotos → YuNet+SFace (`models/face/`, incluidos en el repo) generan un embedding → la cara queda en lista prohibida. El monitor periódico bloquea la estación de trabajo (Win+L) tras N avisos consecutivos de una cara baneada. Acciones `faceguard.ban/status/remove/monitor.start/monitor.stop`; ban y remove son SENSITIVE.
- **Generación local de imágenes — Qwen-Image-2.1-Turbo**: texto perfectamente legible (el punto débil de SDXL). Ciclo de vida secuencial diseñado para 8GB VRAM/15GB RAM: el encoder Qwen3-VL codifica el prompt en CPU y se libera, el transformer (cuantizado a FP8 rowwise con torchao, 3.6GB) y la VAE decodifican en GPU. En la máquina de desarrollo es el modelo por defecto (`JARVIS_IMAGE_MODEL_KIND=qwen_turbo`); el default del código sigue siendo `sdxl` y cae automáticamente a SDXL si faltan los ~24GB de modelos. La imagen online usa `gemini-2.5-flash-image` (tier gratis) con fallback local.
- **SDXL 28× más rápido en 8GB**: detección automática de VRAM libre — con <10GB el runtime usa `enable_model_cpu_offload` + VAE tiling (2.8 s/paso a 1024px; antes 78.8 s/paso thrasheando a RAM).
- **VRAM swap**: `VramSwapGuard` detiene el LLM local antes de cargar modelos pesados y lo regresa después, respetando ownership externa.
- **Voz**: edge-tts como proveedor por defecto (voz Álvaro, español natural, cero latencia de red), XTTS con re-clonación pendiente, fallbacks en cascada; los textos largos se segmentan y se lee markdown limpio (sin asteriscos).
- **Estabilidad de suite/CI**: importorskip en tests que requieren GPU (CI no instala extras de imagen), pin de `transformers` eliminado del workflow, 1366 tests verdes con la configuración real de la máquina activa, CI Windows verde.

### 2026-08-16 — Nemotron híbrido, adjuntos multimodales y estabilización

- El modelo principal se migró desde los proveedores anteriores a `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` mediante NVIDIA API y al objetivo local exacto `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` mediante un endpoint OpenAI-compatible. Las integraciones de modelo anteriores se retiraron del código activo, UI, CLI, configuración y pruebas; no existe un fallback oculto fuera de los dos proveedores Nemotron.
- Se añadieron los modos persistentes `AUTO`, `NVIDIA API` y `LOCAL`. AUTO comprueba el endpoint NVIDIA con autenticación, usa el proveedor local ante clave ausente/inválida, `429`, timeout, red o `5xx`, y vuelve a probar NVIDIA después del enfriamiento. Los modos forzados nunca cambian silenciosamente de proveedor.
- Se implementó un circuit breaker de tres fallos, respeto de `Retry-After`, estados observables, proveedor fijo por solicitud y recuperación de streaming sin repetir efectos externos ni llamadas de herramientas.
- La interfaz muestra modo, proveedor, modelo, salud, latencia, motivo de fallback, estado de clave y capacidades. Ajustes permite guardar, reemplazar, eliminar y probar la clave mediante el almacén seguro del sistema; el frontend solo recibe estados redactados.
- El compositor incorpora un botón `+`, selección múltiple, arrastrar y soltar, pegado, navegación de teclado, chips de vista previa, cancelación, eliminación y reintento. Los adjuntos pendientes sobreviven al cambio de modo.
- Se creó un pipeline neutral y acotado por solicitud para imágenes, audio, video, PDF, DOCX, TXT, Markdown, CSV, JSON y código. Valida magic bytes, MIME, nombres, tamaño, corrupción, rutas de archivo y contenido sensible; almacena referencias protegidas en lugar de Base64 en el historial.
- Se conectaron OCR local, transcripción con segmentos, extracción de fotogramas/audio de video y extracción fragmentada de documentos. `pypdf` es ahora una dependencia base para PDF y `keyring` pertenece al extra desktop.
- Se añadió diagnóstico de RAM, GPU/VRAM, driver, CUDA, backend, NVFP4, contexto y paralelismo. El launcher vLLM conserva el checkpoint y la cuantización exactos y solo pasa offload/límites que el backend declara soportar.
- El chequeo local de `/models` ahora exige que el endpoint anuncie el identificador NVFP4 exacto; una respuesta JSON válida que sirve otro modelo se rechaza con diagnóstico de incompatibilidad, sin sustitución silenciosa.
- El extra `dev` incluye ahora `build` y `wheel`, además de pytest y Ruff, para que el comando de construcción documentado sea reproducible en un entorno nuevo.
- Se corrigió el cierre de logging: Jarvis ya no llama `logging.shutdown()` global ni cierra handlers del host/pytest; solo descarga y cierra sus propios handlers, tolerando errores de cierre independientes.
- El estado de voz dejó de cargar `pyttsx3` durante cada refresco y la síntesis/reproducción queda serializada para evitar carreras entre respuestas. El watchdog renueva el deadline de operaciones que emiten heartbeat real, pero sigue cancelando tareas estancadas.
- El indexado de arranque/eventos/autonomía se ejecuta en una cola de fondo de un trabajador; se añadieron pausa, reanudación y cancelación pendiente mediante API y `jarvis index`, mientras las consultas de chat evitan esperar detrás de esa cola.
- La investigación registra etapas separadas de memoria semántica, memoria, workspace y web. Una colección vacía finaliza explícitamente y un fallo de síntesis entrega un informe parcial con las fuentes recuperadas en vez de perder la evidencia.
- La aplicación desktop muestra una superficie de inicio antes de construir el runtime pesado. Los benchmarks rápidos se identifican como simulados y ya no inventan throughput; los benchmarks reales conservan proveedor, modelo, TTFT, streaming y errores observables.
- Se corrigieron rutas de request que perdían adjuntos, contratos HTTP que descartaban `tools`, `tool_choice` o `response_format`, estados de clave que no reflejaban validación y textos de UI que anunciaban un modelo retirado.
- Se corrigieron estados híbridos que quedaban obsoletos al cambiar de modo, timestamps inestables en snapshots inactivos y lecturas de modo que creaban configuración incidentalmente y contaminaban el indexado.
- Se conservó la compatibilidad con handlers de chat que aceptan únicamente el argumento histórico, el envío de UI se agenda en el siguiente ciclo de eventos sin una demora artificial de 25 ms y cada stream conserva sus IDs de request/turno para impedir repeticiones.
- La configuración persistente pasó a rutas de usuario mediante `platformdirs`, escritura atómica y redacción de campos sensibles; la primera lectura puede migrar una configuración de workspace existente sin reescribirla ni crear archivos incidentalmente.
- `python -m jarvis doctor` ahora incluye modo híbrido persistido, ambos modelos exactos, capacidades, diagnóstico de hardware local y estado redactado de la clave; ya no informa falsamente que todo LLM en línea está desactivado.
- Se añadieron pruebas de regresión para selección/failover/recuperación, secreto, proveedor local/remoto, streaming, tool calling, salida estructurada, adjuntos y seguridad multimodal, UI, logging, diagnóstico y migración. La prueba NVIDIA real es opt-in mediante `RUN_NVIDIA_LIVE_TESTS=true`.
- La compatibilidad del protocolo local se probó además contra un servidor HTTP OpenAI-compatible real en `127.0.0.1` (salud, inferencia, streaming, herramientas y salida estructurada); las pruebas normales no necesitan red externa.

### 2026-08-15 — Auditoría técnica y estabilización

- La validación de seguridad para una ventana desconocida ahora consulta únicamente el catálogo de aplicaciones ya almacenado en memoria. Ya no inicia un escaneo recursivo del sistema durante una confirmación interactiva, evitando bloqueos de varios minutos sin debilitar la confirmación obligatoria.
- Coqui XTTS valida la muestra `speaker_wav` antes de cargar el modelo. Una solicitud inválida degrada inmediatamente al siguiente proveedor configurado y no reserva memoria para un modelo que no podrá sintetizar.
- Una preferencia explícita por CUDA se rechaza cuando `torch.cuda.is_available()` es falso. El proveedor XTTS se declara no saludable y el enrutador utiliza el orden de fallback en vez de intentar ejecutar una compilación CPU de PyTorch sobre CUDA.
- La reproducción de voz prueba backends compatibles en orden. Si el backend preferido falla, utiliza el siguiente disponible y conserva `playback_fallback_from` y `playback_fallback_reason` en los metadatos del resultado.
- Las pruebas automatizadas aíslan los backends opcionales de audio mediante proveedores `in_memory`; las pruebas específicas de voz siguen activando explícitamente XTTS y otros proveedores cuando corresponde. Esto evita depender del hardware o de la `.env` personal durante la batería general.
- Se añadieron regresiones para el catálogo de aplicaciones, la validación temprana de XTTS, CUDA no disponible y el fallback del backend de reproducción.
- Se incorporó Ruff al extra `dev` con una configuración reproducible para errores de sintaxis, imports y nombres. La limpieza inicial eliminó importaciones y variables muertas sin alterar las API públicas reexportadas desde los paquetes.

## Source Tree

Árbol condensado por subsistemas (`src/jarvis/`):

```text
src/jarvis/
  __main__.py  bootstrap.py  cli.py  config.py  identity.py
  desktop.py  jarvis_logging.py  persistent_config.py  reasoning.py
  capability_manifest.py  legacy_model_config.py  runtime_paths.py
  # núcleo
  core/            # errors, events, modes, safety, services, state
  actions/         # contratos de acciones, registro, receipts, routing
  tools/           # tool calling de bajo nivel
  routing/         # task router: metacommands / tools / actions / orquestación
  cognition/       # kernel de orquestación intent→plan
  conversation_context/  conversational_safety.py  chat_optimization.py
  # modelos
  models_runtime/  # nvidia_nim, qwen_local, nemotron_local, hybrid, router,
                   # local_profiles, model_identity, roles, adapters
  local_runtime/   # diagnóstico de hardware y lanzadores de backend
  neural_router.py  local_model_ab.py
  # runtimes de capacidad
  voice_runtime/        # TTS (edge-tts, XTTS, pyttsx3), STT, segmentación,
                        # gemini_live (conversación en tiempo real)
  browser_runtime/      # Playwright: sesión persistente, a11y [N], descargas,
                        # back/find, tabs
  image_runtime/        # Qwen-Image-2.1-Turbo (qwen_backend, secuencial) y
                        # SDXL (DiffusersImageBackend) + Gemini online
  face_guard/           # YuNet+SFace, lista de caras prohibidas, monitor+lock
  vision_runtime/       terminal_runtime/  system_runtime/  windows_runtime/
  file_runtime/  indexing_runtime/  research_runtime/  writing_runtime/
  science_runtime/  data_analyzer/  calculator_runtime.py  hud_runtime/
  # agentic y automatización
  agent_runtime/  desktop_agent_runtime/  code_agent_runtime/
  skills_runtime/  self_improvement_runtime/  automation/  autonomy/
  security_runtime/  memory/  memory_semantic/  web_search/  ingestion/
  integrations/  ops/  services/  modules/  ui_automation/  unity_runtime/
  # superficies
  api/app.py  desktop.py  benchmark.py
```

Los tests viven en `tests/` (188 archivos); los modelos en `models/`
(`llm/` GGUF del LLM local, `image/` checkpoint SDXL, `qwen_image_turbo/`
pipeline Qwen, `face/` ONNX YuNet+SFace — los últimos incluidos en el repo).

## Core Dependencies

- `pydantic` and `pydantic-settings`: shared contracts, typed payload validation, and centralized configuration.
- `SQLAlchemy`: local persistence for memories, activity records, and automation state.
- `APScheduler`: recurring execution for background automations.
- `FastAPI`: local HTTP service surface for a future desktop shell or local agents.
- `Typer`: operational CLI on top of the same runtime service facade.
- `Pillow`: base image inspection capability for the vision layer.
- `pypdf`: extracción de texto y referencias de página de documentos PDF.
- `keyring` (extra `desktop`): almacenamiento de la clave NVIDIA en el almacén seguro del sistema operativo.

## Runtime Layers

- `config.py`: central settings, environment preparation, paths, logging, and safety limits.
- `core/`: errors, events, runtime state, mode policy, metacommand parsing, and service contracts.
- `models/`: common strict Pydantic base models.
- `actions/`: executable action contracts, validation, receipts, and rollback-aware routing.
- `tools/`: lower-level tool calling layer for direct capability access.
- `models_runtime/`: contratos neutrales, los dos proveedores Nemotron, coordinación híbrida y circuit breaker.
- `attachments/`: detección segura, almacenamiento opaco y procesamiento multimodal limitado a la solicitud.
- `local_runtime/`: diagnóstico de hardware y adaptadores de lanzamiento extensibles para backends NVFP4 compatibles.
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
6. `ModelService` sits behind `ProviderRegistry`, `ModelRouter` and the hybrid coordinator, so UI, memory, tools, actions and orchestration never couple themselves to NVIDIA, vLLM or another backend.
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

El selector del encabezado permite `AUTO`, `NVIDIA API` y `LOCAL`. El cambio se guarda en la configuración persistente, se aplica a la siguiente solicitud y no interrumpe el turno que ya está en curso. La UI muestra qué proveedor atendió cada respuesta; no confunde conectividad general con disponibilidad autenticada de NVIDIA.

Si deseas diagnosticar el estado del entorno de red y los modelos detectados, usa el comando de diagnóstico:

```powershell
python -m jarvis doctor
```
o su equivalente para el Code Agent:
```powershell
python -m jarvis code doctor
```

## Modelo principal híbrido

Los dos proveedores nuevos entregan el mismo `ModelResponse` interno:

- `NvidiaNimProvider`: endpoint NVIDIA NIM autenticado, modelo remoto fijo `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`.
- `NemotronOmniLocalProvider`: endpoint OpenAI-compatible configurable, modelo fijo `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`.

`selectedMode` conserva la elección del usuario; `requestProvider` queda fijado al comenzar el turno; `activeProvider` identifica el último proveedor efectivo; `providerStatus`, `capabilities` y `fallbackReason` explican la decisión sin exponer secretos.

| Modo | Comportamiento |
| --- | --- |
| `AUTO` | Prefiere NVIDIA si la clave existe y `/models` responde autenticado. Ante `401/403`, `429`, timeout, red o `5xx`, usa Nemotron local cuando la modalidad tenga una ruta segura. Reevalúa NVIDIA en la siguiente solicitud después del enfriamiento. |
| `NVIDIA API` | Fuerza NVIDIA. Si falla, devuelve un error claro y no cambia al modelo local. |
| `LOCAL` | Usa exclusivamente `LOCAL_LLM_BASE_URL`; no realiza solicitudes a NVIDIA. Si el endpoint se aloja en otra máquina, solo se contacta esa URL configurada. |

El circuit breaker abre aproximadamente tras tres fallos consecutivos. Un `429` respeta `Retry-After`. Si un stream remoto falla antes de una acción externa, AUTO puede reintentarlo localmente una sola vez; si ya hubo un efecto, el turno se marca incompleto y requiere confirmación para repetirlo. Los identificadores de turno, request, archivo y tool call evitan duplicados.

### Configuración y ubicación de la clave

Copia `.env.example` a un archivo local `.env`, que ya está ignorado por Git:

```dotenv
JARVIS_LLM_MODE=auto
NVIDIA_API_KEY=TU_CLAVE_REAL
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning

LOCAL_LLM_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_LLM_MODEL=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
LOCAL_CPU_OFFLOAD_GB=auto
LOCAL_GPU_MEMORY_UTILIZATION=auto
LOCAL_MAX_MODEL_LEN=auto
LOCAL_TENSOR_PARALLEL_SIZE=auto
JARVIS_LOCAL_LOAD_TIMEOUT_SECONDS=900
JARVIS_LOCAL_PERFORMANCE_PROFILE=stable
```

La clave real va exactamente en `NVIDIA_API_KEY` dentro de `.env`, o en **Ajustes → NVIDIA → API key**. El campo es enmascarado y guarda mediante el keyring del sistema cuando está disponible. No se conserva en `localStorage`, logs, conversación, telemetría, archivos del frontend ni respuestas HTTP. El endpoint `/llm/status` solo indica `configured`, `source` y `validity`; `/llm/nvidia-key` nunca devuelve el valor. Si no existe un keyring utilizable, la UI indica que debe usarse `.env`.

### Iniciar Nemotron NVFP4 local

Instala un backend que soporte realmente este formato en hardware compatible y sirve el checkpoint exacto. El adaptador incluido genera una orden equivalente a:

```powershell
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 `
  --served-model-name nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 `
  --host 127.0.0.1 --port 8000 `
  --trust-remote-code `
  --reasoning-parser nemotron_v3 `
  --enable-auto-tool-choice `
  --tool-call-parser qwen3_coder `
  --kv-cache-dtype fp8
```

Si `LOCAL_TENSOR_PARALLEL_SIZE` es mayor que uno, el adaptador añade el backend MoE indicado por la receta oficial. La orden solo se construye cuando el diagnóstico confirma vLLM 0.20.0 o superior, CUDA 12.9 o superior y una GPU Blackwell compatible; no se descarga ni intenta cargar el checkpoint en un host incompatible.

Después selecciona `LOCAL` o deja `AUTO`. También puedes establecer `LOCAL_LLM_BASE_URL=http://servidor-compatible:8000/v1` para alojarlo en otra máquina. Jarvis no cambia automáticamente a otro modelo ni a otra cuantización.

Los valores `auto` delegan los límites al backend, evitando inventar una configuración segura. Para un perfil explícito, establece `JARVIS_LOCAL_PERFORMANCE_PROFILE=custom` y solo los límites que el backend soporte, por ejemplo:

```dotenv
LOCAL_CPU_OFFLOAD_GB=12
LOCAL_GPU_MEMORY_UTILIZATION=0.82
LOCAL_MAX_MODEL_LEN=32768
LOCAL_TENSOR_PARALLEL_SIZE=2
JARVIS_LOCAL_LOAD_TIMEOUT_SECONDS=1200
JARVIS_LOCAL_PERFORMANCE_PROFILE=custom
```

vLLM declara soporte para esas cuatro opciones; un adaptador futuro que no soporte alguna es rechazado con el campo exacto, no ignorado. Los perfiles `stable`, `balanced` y `performance` sirven como etiqueta operativa configurable; los límites numéricos siguen siendo explícitos o `auto`, de modo que el nombre del perfil nunca inventa VRAM/RAM disponible.

### Compatibilidad del equipo auditado

El equipo usado para esta auditoría tiene aproximadamente 13.9 GiB de RAM, PyTorch 2.11 CPU-only, CUDA no disponible, ningún GPU NVFP4 verificable y vLLM ausente. El diagnóstico local rechaza el lanzamiento porque NVFP4 requiere NVIDIA Blackwell con compute capability 10.0 o superior, CUDA/backend compatible y memoria suficiente; el repositorio oficial del checkpoint publica aproximadamente 22.4 GB de archivos. Por tanto, **NVFP4 no fue ejecutado en este equipo**. La arquitectura, endpoint, launcher y failover sí se verifican con mocks y un servidor HTTP OpenAI-compatible real, y quedan listos para hardware compatible o un host externo.

## Research System V2: Web y archivos compartidos

`web.search`, `files.search`, `files.search_content` y `files.read_text` se
registran en la capa común de acciones. El modelo local y el online reciben el
mismo registro y sólo el subconjunto relevante para cada petición.

Configura el `.env` situado en la raíz del repositorio:

```dotenv
BRAVE_SEARCH_API_KEY=tu_clave
YDC_API_KEY=tu_clave_you
EXA_API_KEY=tu_clave_exa
SEARXNG_BASE_URL=http://127.0.0.1:8080
JARVIS_WEB_SEARCH_PROVIDER=brave
JARVIS_WEB_SEARCH_ENABLED=true
```

`BRAVE_SEARCH_API_KEY` es el nombre canónico. Por compatibilidad también se
aceptan `JARVIS_BRAVE_SEARCH_API_KEY` y `BRAVE_API_KEY`. El archivo `.env` se
resuelve desde la raíz del proyecto, aunque JARVIS se inicie desde otra carpeta.

Comprobaciones seguras:

```powershell
python -m jarvis web status
python -m jarvis web health
python -m jarvis web search "OpenAI" --max-results 1
```

`status` no consume peticiones y muestra You, Brave, Exa y SearXNG por separado;
`health` hace una consulta mínima por proveedor configurado. You es el buscador
general preferido, Exa aporta recuperación semántica/técnica y highlights,
Brave se conserva como proveedor completo y SearXNG es cobertura local opcional.
El router elige por intención, disponibilidad y profundidad; un fallo temporal
no desactiva permanentemente ninguna herramienta.

### Web Search y Deep Research

JARVIS separa `ResearchDepth` (`NONE`, `NORMAL`, `DEEP`) de `SourceScope`
(`AUTO`, `FILES`, `WEB`, `FILES_AND_WEB`). Una petición explícita a archivos
nunca llama a Web ni cambia silenciosamente de fuente; una petición combinada
busca ambos lados, conserva procedencia y jamás incorpora fragmentos privados a
las consultas públicas.

JARVIS normaliza, deduplica y ordena
los candidatos, abre un conjunto diverso de páginas públicas, extrae fragmentos
relevantes y conserva la relación entre evidencia, fuente y cita. El mismo
orquestador se usa con el modelo local y con el online; el LLM sintetiza la
evidencia, pero no controla la búsqueda ni puede inventar URLs válidas.

`auto` elige `no_web`, `quick` o `normal` según intención y freshness; nunca
activa `deep` por sí solo. Usa el botón **Deep Research** o
`/deepresearch <pregunta>` para el presupuesto amplio y adaptativo. Durante la
ejecución la UI muestra únicamente actividad operacional y cifras reales; Stop
cancela rondas y peticiones pendientes de forma cooperativa.

```powershell
python -m jarvis web search "OpenAI latest" --mode normal
python -m jarvis web research "current small multimodal open-source models" --deep
```

Los límites centrales por defecto son techos de seguridad, no objetivos de
calidad: 1/3/10 consultas para
quick/normal/deep, 3/6/12 páginas, 8 resultados por consulta, 80 candidatos
totales, 36 fragmentos, 12 000 tokens de contexto, 3 rondas, concurrencia 3/4
para consultas/lecturas, timeout de 12 s y cachés de 120/600 s. Todas las
opciones aparecen en `.env.example` con el prefijo `JARVIS_SEARCH_`. La
investigación se detiene por cobertura de requisitos, contradicciones e
información nueva; si llega a un techo sin cobertura, informa estado parcial.

SearXNG no es requisito de arranque. Una instalación local reproducible está
documentada en `docs/searxng.md`; JARVIS lo detecta únicamente cuando
`SEARXNG_BASE_URL` está configurado y el health check responde.

La síntesis reserva 1 800 tokens de salida para Deep Research (1 200 en
búsqueda normal), desactiva razonamiento oculto durante la redacción final y
permite como máximo dos continuaciones internas. Una respuesta cortada se
fusiona y deduplica antes de crear la única burbuja final. Si las citas fallan,
se intenta una reparación y luego una síntesis compacta; los tres intentos son
el límite absoluto y todos vuelven a pasar por validación de IDs y URLs.

El lector sólo permite HTTP(S) público, valida DNS y cada redirección, bloquea
localhost/LAN/link-local/metadata/credenciales, limita bytes y tipos de
contenido, y marca cada pasaje como no confiable. Las pruebas unitarias bloquean
tráfico externo por defecto; las pruebas reales viven en `tests/integration/` y
sólo se activan con `RUN_WEB_RESEARCH_LIVE_TESTS=1` o
`RUN_WEB_RESEARCH_E2E=1`. La regresión médica general del fallo de síntesis se
activa aparte con `RUN_CANCER_DEEP_RESEARCH_E2E=1`.

El agente obtiene sus capacidades ejecutables del registro central de acciones,
no de una lista aprendida por el modelo. Local y online reciben la misma fuente
de verdad; cuando el catálogo crece se seleccionan schemas relevantes para la
consulta. El manifiesto visible agrupa nombres reales como `files.search`,
`files.read_text`, `web.search` y la capacidad orquestada
`web.deep_research`.

## Adjuntos multimodales (`+`)

El botón `+` está junto al campo de chat y abre categorías para imágenes, video, audio, documentos, texto, código y otros archivos seguros. Admite selección múltiple, arrastrar/soltar, pegar desde el portapapeles, teclado (`Enter`/`Space` para abrir y `Esc` para cerrar), vista previa, nombre/formato/tamaño/estado, cancelar, eliminar y reintentar. Puedes enviar texto junto con varios archivos; cambiar de modo no borra la cola.

Cada archivo se representa mediante ID, nombre saneado, categoría, MIME detectado, tamaño, SHA-256, token local opaco, estado, modalidad y contenido derivado. El historial no recibe Base64 ni rutas reales. Por defecto se permiten 16 archivos, 50 MiB por archivo y 200 MiB por request; se configuran con `JARVIS_ATTACHMENT_MAX_FILES_PER_REQUEST`, `JARVIS_ATTACHMENT_MAX_FILE_BYTES` y `JARVIS_ATTACHMENT_MAX_REQUEST_BYTES`.

Rutas de procesamiento:

- Imágenes: transporte nativo cuando el proveedor y su protocolo lo verifican; en los demás casos OCR/visión local. Se conserva el original protegido.
- Audio: transporte nativo solo si está verificado; de lo contrario FFmpeg normaliza y el STT local devuelve texto y segmentos temporales.
- Video: FFprobe/FFmpeg extraen fotogramas acotados, audio y timestamps; el pipeline transcribe y construye contenido derivado sin enviar el video completo innecesariamente.
- Documentos: TXT, Markdown, CSV, JSON, código, DOCX y PDF se extraen localmente, se fragmentan con presupuesto de contexto y conservan referencias de página cuando existen. Un PDF escaneado puede continuar por OCR local.

Se rechazan path traversal, nombres absolutos, archivos ejecutables, HTML/SVG activo, MIME falso, PDF/DOCX/JSON corruptos, archivos excesivos y patrones de secretos. El almacenamiento queda limitado al request actual. `LOCAL` procesa todo mediante rutas locales y garantiza cero tráfico a NVIDIA; NVIDIA API recibe solo el archivo adjunto o contenido derivado mínimo que la modalidad requiera. Los adjuntos no entran en memoria permanente sin una política explícita.

## Capacidades conservadas y adaptadas

La migración es aditiva respecto de las capacidades del sistema. Se mantienen memoria SQL/local, conversación, voz/STT/TTS, visión/OCR, generación de imágenes, automatizaciones, razonamiento, lectura/ingesta de documentos, búsqueda Brave, Code Agent, Git/repo learning, Guided Control, herramientas y confirmaciones de seguridad. Las herramientas se ejecutan localmente bajo las mismas políticas: Nemotron puede proponer una llamada, pero no saltarse permisos ni confirmaciones.

Nemotron se aprovecha mediante streaming, razonamiento, tool calling y salida JSON normalizada. La ficha oficial declara texto, imagen, audio y video como entradas y un contexto de hasta 256K, pero el transporte NVIDIA implementado solo marca imagen como modalidad binaria nativa verificada. Audio/video usan el pipeline local derivado hasta verificar un contrato de transporte estable; documentos crudos no se anuncian como modalidad nativa.

## Diagnóstico y solución de problemas

- `cloud_invalid_key`: guarda una clave nueva en Ajustes o corrige `NVIDIA_API_KEY`; AUTO puede continuar localmente, NVIDIA API no.
- `cloud_rate_limited`: se muestra el enfriamiento de `Retry-After`; AUTO usa local temporalmente.
- `cloud_unavailable` o timeout/`5xx`: verifica `NVIDIA_BASE_URL`; AUTO abre el circuito tras fallos repetidos.
- `local_active` con error de conexión: inicia el endpoint exacto o corrige `LOCAL_LLM_BASE_URL`. Ajustes muestra el diagnóstico y la orden de lanzamiento solo cuando el host es compatible.
- `both_unavailable`: revisa conjuntamente la clave/servicio NVIDIA y el endpoint local; Jarvis muestra el diagnóstico combinado y no cambia a un tercer proveedor.
- OCR, audio o video no disponibles: instala/configura Tesseract, FFmpeg/FFprobe y el STT local. El adjunto queda en error reintentable con el requisito faltante.
- El archivo se rechaza antes de subir: revisa tamaño, extensión/contenido, integridad y que no incluya credenciales. El mensaje identifica la validación que falló.

### Control del indexado en segundo plano

```powershell
python -m jarvis index status
python -m jarvis index pause
python -m jarvis index resume
python -m jarvis index cancel-pending
python -m jarvis index run
```

`pause` impide que comiencen nuevos trabajos de fondo sin interrumpir una consulta de chat en primer plano; `cancel-pending` cancela solo los trabajos que aún no empezaron. El estado expone `background_active`, `background_paused` y la política de prioridad de chat.

## Verificación

Comandos reproducibles usados durante esta migración:

```powershell
python -m pip check
python -m ruff check --no-cache .
python -m ruff format --check --no-cache <archivos Python nuevos de esta migración>
$env:QT_QPA_PLATFORM='windows'
python -m pytest -p no:cacheprovider -q
python -m build --no-isolation
python -m jarvis doctor
python -m jarvis serve
python -m jarvis.desktop
```

Resultados verificados el 2026-10-25 (round navegador/live/guardián/Qwen):

- Batería local completa con la `.env` real de la máquina activa: `1366 passed, 5 skipped` en ~5 min.
- CI de GitHub (Windows runner) verde en `59ee522`: ruff + batería completa con entorno recién instalado.
- E2E reales verificados en la laptop (RTX 5070 8GB / 15GB RAM): generación Qwen-Image-2.1 con texto legible, arranque del modo live con micrófono, navegador abriendo YouTube por comando de voz del modelo local, guardián de caras con webcam (detección honesta sin cara presente).
- Benchmarks medidos: SDXL offload 2.8 s/paso @1024px; Qwen secuencial ~23 min/imagen (FP8 sin triton en Windows).

Resultados históricos del 2026-08-16 (migración Nemotron):

- Línea base anterior a la migración: `632 passed, 2 skipped`.
- Primera batería completa de cierre: `720 passed, 3 failed, 3 skipped` en 1171.55 s. Los tres fallos reprodujeron contaminación del modo persistido en dos pruebas y la pérdida del encabezado/política del benchmark; sus causas se corrigieron y las cuatro regresiones focalizadas pasaron.
- Regresión dirigida del núcleo híbrido final: `72 passed` en 39.80 s.
- Batería completa final: `724 passed, 3 skipped` en 1869.88 s. Los omitidos son la prueba NVIDIA live (requiere `RUN_NVIDIA_LIVE_TESTS=true`) y dos pruebas de seguridad de symlinks que Windows no permite crear en este entorno.
- Ruff lint: `All checks passed`; los 43 archivos Python nuevos pasaron `ruff format --check`. No se reformatearon masivamente archivos heredados.
- Dependencias: `python -m pip check` no detectó requisitos rotos.
- Build: wheel `jarvis_cognitive_os-0.1.0-py3-none-any.whl` y `sdist` construidos correctamente con `python -m build --no-isolation` desde una copia limpia del mismo árbol fuente.
- Instalación del artefacto: el wheel se instaló con `pip --target` en un directorio aislado, importó `jarvis` y conservó los dos nombres exactos de modelo.
- Smoke API: startup y shutdown limpios; `/health`, `/llm/status` y `/llm/local-diagnostic` respondieron HTTP 200.
- Smoke CLI: `python -m jarvis smoke-test` completó una operación de memoria sobre datos temporales y cerró limpiamente.
- Smoke desktop Qt sobre la plataforma Windows: `3 passed`; mostró la ventana y verificó selector, botón `+`, teclado, multiarchivo, chips, cambio de modo, drop y portapapeles.

El proyecto no define actualmente configuración ni dependencia oficial para mypy o pyright, por lo que no existe un comando de tipos estático reproducible que ejecutar. `compileall` sí aprobó todo `src/jarvis`; los contratos nuevos usan Pydantic estricto y están cubiertos por las pruebas. Añadir un type checker global requeriría una migración separada del código heredado y no se simuló como verificación aprobada.

El workspace de esta sesión impide que procesos secundarios creen o actualicen ciertos archivos dentro del árbol: el build directo no pudo tocar `src/jarvis_cognitive_os.egg-info` y Ruff no pudo reescribir algunos archivos aunque sí pudo verificarlos. El mismo código se construyó y arrancó sin errores al dirigir build/datos a un directorio temporal escribible. La caché de pytest se desactiva por la misma restricción; `-p no:cacheprovider` no omite pruebas. La inspección visual asistida de Windows tampoco estuvo disponible porque el plugin instalado solicita un submódulo no exportado por su runtime; las pruebas Qt nativas con ventana mostrada sí pasaron. Una instalación normal del repositorio debe ejecutarse desde un directorio donde el usuario tenga escritura efectiva.

## Voz y degradación controlada

El runtime de voz selecciona un proveedor TTS y un backend de salida de forma independiente. El proveedor por defecto es **edge-tts** (voz `es-MX-AlvaroNeural`, español natural sin latencia de red ni descargas de modelos); XTTS queda disponible para clonación de voz con una muestra `speaker_wav`, y `in_memory`/`pyttsx3` siguen de respaldo. Los textos largos se segmentan en frases y se reproducen en orden, y el markdown se lee limpio (los `**negritas**` no se verbalizan con asteriscos).

Configuración principal en `.env`:

```dotenv
JARVIS_VOICE_ENABLED=true
JARVIS_VOICE_TTS_PROVIDER_DEFAULT=edge_tts
JARVIS_VOICE_TTS_PROVIDER_FALLBACK_ORDER=coqui_xtts,pyttsx3,in_memory
JARVIS_VOICE_AUDIO_OUTPUT_BACKEND_DEFAULT=winsound
JARVIS_VOICE_COQUI_DEVICE_PREFERENCE=auto
JARVIS_VOICE_COQUI_SPEAKER_WAV=C:/ruta/a/speaker.wav
```

Usa `auto` para seleccionar CUDA sólo cuando PyTorch confirme que está disponible. Si configuras `cuda` explícitamente en un equipo CPU-only, XTTS se marcará como no saludable y Jarvis continuará con el orden de fallback. La muestra WAV debe existir antes de que XTTS cargue el modelo.

### Conversación en tiempo real (`voice.live`)

Con una `GEMINI_API_KEY` configurada, **"Jarvis, modo live"** abre una
conversación por voz bidireccional con latencia de conversación humana
(Gemini Live sobre WebSocket): hablas, te escucha y responde por audio
en el mismo turno, con el cierre de turno por silencio (VAD del
servidor). Las acciones `voice.live.start` / `voice.live.stop` (alias en
español incluidos) la controlan desde el chat o la voz. El LLM local
queda liberado mientras dura el modo.

Comandos útiles:

```powershell
python -m jarvis voice status
python -m jarvis voice speak "Sistema en línea"
python scripts/jarvis_live.py   # launcher directo del modo live
python -m jarvis doctor
```

### Guardián de caras prohibidas

"Jarvis, banea esa cara" captura a la persona frente a la webcam, genera
su embedding con YuNet+SFace (`models/face/`, incluidos en el repo) y la
agrega a la lista prohibida. El monitor periódico bloquea la sesión de
Windows (Win+L) cuando detecta una cara baneada de forma consecutiva.
Acciones: `faceguard.ban` y `faceguard.remove` (SENSITIVE),
`faceguard.status`, `faceguard.monitor.start/stop` (SAFE). Activable con
`JARVIS_FACE_GUARD_ENABLED=true`.

## Local Image Runtime

Jarvis genera imágenes localmente con dos motores intercambiables mediante
`JARVIS_IMAGE_MODEL_KIND`:

| Kind | Modelo | Punto fuerte | Hardware |
| --- | --- | --- | --- |
| `qwen_turbo` | Qwen-Image-2.1-Turbo (7B, 8 pasos, FP8) | Texto perfectamente legible, infografías, rótulos | 8GB VRAM (ciclo secuencial), ~24GB en disco |
| `sdxl` (default del código) | JuggernautXL v8 via Diffusers | Velocidad, fotorrealismo | 8GB VRAM con offload automático |

Qwen usa un **ciclo de vida secuencial** pensado para laptops de 8GB VRAM:
el encoder Qwen3-VL codifica el prompt en CPU y se libera (~9GB), después el
transformer — cuantizado a FP8 rowwise con torchao (3.6GB) — y la VAE
trabajan en GPU y se liberan al terminar. Nunca hay dos componentes pesados
residentes a la vez. Con `qwen_turbo` activo y sin modelos en disco, el
servicio cae a SDXL con un warning en vez de morir.

SDXL detecta la VRAM libre: con <10GB usa `enable_model_cpu_offload` + VAE
tiling automáticamente (2.8 s/paso a 1024px en una RTX 5070 Laptop, ~28× más
rápido que el `.to(cuda)` ingenuo que thrashea a RAM).

La imagen **online** usa `gemini-2.5-flash-image` (tier gratis de Google AI
Studio vía `GEMINI_API_KEY`) con fallback al motor local.

Rutas esperadas:

```text
models/image/checkpoints/juggernautXL_v8Rundiffusion.safetensors   # SDXL
models/qwen_image_turbo/   # pipeline Qwen (transformer+encoder+VAE, ~24GB)
models/face/              # ONNX del guardián de caras (incluir en repo)
```

Los archivos de modelo grandes quedan ignorados por Git (`.safetensors`,
`.ckpt`, `.bin`, `.pt`, `.pth`, `models/qwen_image_turbo/`). No subas
imágenes generadas, `.env`, claves ni carpetas de salida.

Comandos:

```powershell
python -m jarvis image status
python -m jarvis image generate --prompt "un póster que diga 'JARVIS OS' en letras neón"
python -m jarvis image cancel
python -m jarvis image unload
```

Dependencias de imagen opcionales (diffusers, torch, transformers>=4.57,
accelerate, torchao):

```powershell
python -m pip install -e .[image]
```

La primera generación tarda porque el pipeline se carga de forma diferida; los
modelos permanecen cacheados cuando es posible y se descargan con
`python -m jarvis image unload`.

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
cd C:/ruta/al/repositorio/Jarvis
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[desktop,voice,image]
python -m pip install -e .[dev]
```

Recommended validation commands:

```powershell
python --version
python -c "import jarvis; print('ok')"
python -m compileall src\jarvis
python -m pip check
python -m ruff check --no-cache .
python -m pytest -p no:cacheprovider -q
```

Remove any legacy repo-local virtual environment backups after the root `.venv` is confirmed working. Keeping virtual environments or archived copies inside the indexed workspace will slow down searches, validation, and indexing.

## Registering New Capabilities

New capabilities should enter the runtime as modules, actions, and optionally tools. The recommended pattern is:

1. Create a new module under `src/jarvis/modules/`.
2. Register one or more actions in `register_actions()`.
3. Optionally register tools in `register_tools()`.
4. Add the module to `build_application()`.
5. If the capability needs reasoning, use `context.models` or inject `ModelService`; do not call NVIDIA, vLLM or another backend directly.

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

The current codebase includes a desktop shell plus CLI and HTTP surfaces over the same runtime facade. The remote and local Nemotron providers remain behind the model-provider abstraction, while optional local integrations supply STT/TTS, voice cloning, vision capture, embeddings, image generation, Windows automation, Word COM, and Unity workflows. Optional capability providers must continue to degrade through their registries without bypassing safety confirmation, persistence, cancellation, privacy or observability contracts.
