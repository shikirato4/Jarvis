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
