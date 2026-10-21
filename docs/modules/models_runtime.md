# Models Runtime

Responsibility: expose model providers through stable internal contracts with strict hybrid selection, normalized tool/structured output, safe failover and observability.

Key files:
- `src/jarvis/models_runtime/base.py`
- `src/jarvis/models_runtime/catalog.py`
- `src/jarvis/models_runtime/coordinator.py`
- `src/jarvis/models_runtime/hybrid.py`
- `src/jarvis/models_runtime/openai_compatible.py`
- `src/jarvis/models_runtime/nvidia_nim.py`
- `src/jarvis/models_runtime/nemotron_local.py`
- `src/jarvis/models_runtime/registry.py`
- `src/jarvis/models_runtime/router.py`
- `src/jarvis/models_runtime/service.py`

Providers:

- `NvidiaNimProvider` fixes the official NVIDIA model, authenticates health checks, redacts failures and publishes cloud-specific states.
- `NemotronOmniLocalProvider` fixes the exact NVFP4 model and connects to any configured OpenAI-compatible endpoint, including a compatible host on another machine.
- Both normalize response text, usage, tool calls and streaming chunks. Provider code never executes tools.
- `local_llm_backend` keeps `qwen` as the backwards-compatible default and can select `ollama`, `llama_cpp`, `vllm`, or `openai_compatible`. These endpoint adapters reuse one OpenAI-compatible transport and must target an explicitly configured URL; no LAN scanning is performed. Ollama must expose its `/v1` compatibility endpoint.
- The bundled llama.cpp runtime accepts any GGUF model that the installed llama-server build can load. Local readiness validates the configured server model ID and aliases, not a Qwen family name. Replacing a GGUF at the configured path therefore no longer fails a Qwen-family guard; the runtime still uses the configured alias, context size, and launch flags. A model that needs extra files (for example a vision projector) or model-specific launch/parser flags still needs those configured and supported by llama.cpp.
- `local_runtime_model_path` / `JARVIS_LOCAL_RUNTIME_MODEL_PATH` can point to one GGUF file or to the shared `models/llm` directory. In directory mode, Jarvis discovers the sole top-level model GGUF independent of its filename; projector files named `mmproj-*.gguf` or `mmproj.gguf` are excluded. Keep one main model in that shared directory at a time; multiple candidates fail with a clear configuration error. Multimodal GGUFs can use an explicit `local_runtime_mmproj_path` / `JARVIS_LOCAL_RUNTIME_MMPROJ_PATH`. Without one, Jarvis detects an exact `mmproj-<model-file-stem>.gguf`, `mmproj.gguf`, or the sole `mmproj-*.gguf` sidecar in the same directory. Image capability is enabled only when a projector is found.
- For endpoint adapters, set `local_llm_model` to the model identifier advertised by `/v1/models` and `local_llm_base_url` to the endpoint root ending in `/v1`. Optional `local_llm_capabilities` values (`vision`, `tool_calling`, `structured_output`, `reasoning`) are explicit operator declarations. They default off because API compatibility does not prove model capability. `text` and `streaming` are the conservative adapter defaults.
- Endpoint adapters support `list_models()` and `get_model_info()` on their provider object. Model IDs are discovered from the configured endpoint only; the existing application catalog still controls logical task profiles.
- Optional `llm_models` in the existing `Settings` configuration registers named roles. Supported role names include `primary`, `coder`, `fast`, and `vision`; `primary` can be omitted to retain the current Qwen-backed local primary. `coder` routes coding tasks, `fast` routes classification/summarization tasks, and vision requests search configured models whose `vision` capability is explicitly `supported`. A role falls back through its configured `fallbacks`, then the coordinator's primary route.
- Role capabilities are tri-state (`supported`, `unsupported`, `unknown`). Only `supported` is eligible for capability-specific routing. `text_generation` is inferred from the chat-completions contract; other capabilities stay unknown unless supplied. A configured `context_window_tokens` bounds request budgeting, and requests exceeding that window fail with a typed context-limit error instead of being sent.
- `ModelService` is the internal provider-neutral LLM API: `chat`, `generate`, `stream`, and `for_task`. `discover_models()` is an explicit call and returns provider/model/identifier/version metadata; discovery is not performed during inference. `switch_model(role, model)` validates against the provider's advertised model IDs. A backend that cannot live-load a new ID rejects the switch and requires its normal reload/restart procedure.

Example role configuration (set `JARVIS_LLM_MODELS` to this JSON object):

```json
{
  "coder": {
    "provider": "ollama",
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "qwen2.5-coder:14b",
    "capabilities": {
      "streaming": "supported",
      "tool_calling": "supported",
      "structured_output": "supported"
    },
    "context_window_tokens": 32768,
    "fallbacks": ["primary"]
  },
  "fast": {
    "provider": "ollama",
    "base_url": "http://127.0.0.1:11434/v1",
    "model": "qwen3.5:4b",
    "capabilities": {"streaming": "supported"},
    "fallbacks": ["primary"]
  }
}
```

Example environment configuration for Ollama:

```dotenv
JARVIS_LOCAL_LLM_BACKEND=ollama
JARVIS_LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
JARVIS_LOCAL_LLM_MODEL=ornith-1.5-9b
JARVIS_LOCAL_LLM_CAPABILITIES=["tool_calling","structured_output","reasoning"]
```

Coordinator state is intentionally separate:

- `selected_mode`: the persisted user choice.
- `request_provider`: immutable provider assigned to a request in progress.
- `active_provider` and `active_model`: provider/model that last handled work.
- `provider_status`: health, latency, state, message and retry delay.
- `capabilities`: verified model/transport capabilities.
- `fallback_reason`: explicit cause of an AUTO transition.

The official model card advertises text/image/audio/video input, text output, reasoning, JSON/tool calling and a 256K context. The current NVIDIA OpenAI-compatible adapter verifies binary image transport only; audio/video continue through local derived pipelines until their wire contract is verified. Raw PDF/DOCX are not advertised as native model input.

Failure points:

- missing/invalid NVIDIA key (`401/403`)
- rate limiting (`429`) and `Retry-After`
- timeout, network and `5xx` failures
- an open cloud circuit or both providers unavailable
- local endpoint stopped or serving a model other than the exact NVFP4 target
- unsafe streaming retry after a tool/external effect
- unsupported native modality/transport
- parsing failures on structured output or tool arguments

Required tests cover all modes, persistence, strictness, recovery, no NVIDIA traffic in LOCAL, no duplicate tool effects, model mismatch and secret redaction. NVIDIA live tests remain opt-in with `RUN_NVIDIA_LIVE_TESTS=true`.
