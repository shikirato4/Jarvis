# Architecture

The current Jarvis codebase is centered on a local runtime kernel with explicit boundaries:

- `bootstrap.py` wires all dependencies in one place.
- `core/` contains policy and runtime contracts that other layers depend on.
- `actions/`, `tools/`, `models_runtime/`, `attachments/`, and `local_runtime/` are separate execution surfaces.
- `routing/` is the first entry point for top-level work.
- `services/runtime.py` is the stable facade for external callers.
- `core/capabilities.py` is the central registry for intents, actions, tools, and planning metadata.
- `models_runtime/service.py` is the only application layer that selects model providers.
- `models_runtime/coordinator.py` owns AUTO/NVIDIA API/LOCAL semantics, request pinning, circuit breaking and safe streaming recovery.
- `attachments/` owns request-scoped file validation, protected storage and derived multimodal content.
- `local_runtime/` diagnoses the host and builds commands through backend adapters; adding a backend does not modify `ModelService`.

This split matters because the desktop shell, local automations, background agents, and future multimodal providers should all talk to the same core without duplicating runtime logic.

## Provider-neutral request flow

1. Desktop, CLI or API creates a `ModelRequest`; secret values and raw media are excluded from serialization.
2. `ModelService` delegates every primary LLM request to `ProviderCoordinator`; the catalog/router remain neutral extension seams and contain no retired provider registration.
3. The coordinator reads `selectedMode`, checks the concrete authenticated endpoint required by that mode, fixes `requestProvider` for the turn, and records `activeProvider`, status and fallback reason.
4. `NvidiaNimProvider` and `NemotronOmniLocalProvider` translate OpenAI-compatible payloads into the same `ModelResponse`/`StreamChunk` contracts. Tool calls are proposals only; the local action/tool layers still enforce permissions.
5. Memory, conversation, policies, tools and attachment storage stay provider-neutral and local. Only the minimum prompt or explicitly attached/derived content crosses the selected provider boundary.

## Hybrid invariants

- `AUTO` prefers authenticated NVIDIA NIM and safely degrades only to the exact configured local Nemotron endpoint.
- `NVIDIA API` and `LOCAL` are strict. LOCAL never invokes the NVIDIA provider.
- A provider never changes after output/effects begin. A stream may retry once locally only before any external effect and when the modality has a compatible route.
- `401/403`, `429`/`Retry-After`, timeout/network and `5xx` remain distinguishable. Three consecutive cloud failures open the circuit until cooldown/recovery.
- No retired provider is registered by source, UI, CLI or configuration, and neither AUTO nor either strict mode has a third-provider fallback.
- The local target is always `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`. Hardware incompatibility produces a diagnostic, never a silent model or quantization substitution.

## Multimodal boundary

The desktop composer assigns a request ID before files are accepted. `AttachmentService` validates name, size, content/MIME, integrity and secret patterns, stores bytes under an opaque token, and exposes only a safe descriptor. `MultimodalPipeline` either selects a verified native transport or runs a registered local processor. `request_builder.py` adds bounded derived text and non-serializable media to the model request. Successful send removes the current request's files; failed/cancelled files remain explicit and retryable.

The pipeline is extensible by attachment category. Text/code, JSON, CSV, DOCX and PDF use local extraction/chunking; images use native vision or local OCR; audio uses native transport or local transcription; video uses bounded FFmpeg frames/audio/transcription. Complete files are never stored as Base64 in conversation history.

## Persistence and secrets

SQLAlchemy/SQLite remains the persistence layer for local memory and activity. The selected LLM mode is persisted independently of conversations and pending attachments. NVIDIA credentials are resolved from an OS keyring first and `.env` second; only configured/source/validity status reaches UI or HTTP. Logging owns and closes only handlers it created so embedding hosts and pytest keep their handlers alive.
