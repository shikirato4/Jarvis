# JARVIS Nemotron Runtime Hardening Design

**Date:** 2026-08-16
**Status:** Approved continuation of Option A
**Scope:** Stabilize the existing hybrid Nemotron implementation without replacing its architecture.

## Goals

- Make `AUTO`, `NVIDIA API`, and `LOCAL` the only active LLM modes.
- Remove every active GPT-OSS and Ollama route while preserving an idempotent configuration migration to `AUTO`.
- Keep the exact remote model `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` and exact local target `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`.
- Preserve all existing JARVIS capabilities and user data.
- Keep the UI responsive while chat, speech, indexing, health checks, research, and attachment processing run.
- Move mutable runtime state out of the source tree and make persistence and logging safe across restarts.

## Provider and request invariants

`ModelService` always receives a `ProviderCoordinator`; there is no legacy provider branch. The coordinator owns mode selection and returns a request-scoped provider selection. A request never switches provider after output or an external tool effect has begun.

`AUTO` tries authenticated NVIDIA health and request execution first, then the configured OpenAI-compatible local Nemotron endpoint when failover is safe. `NVIDIA API` and `LOCAL` are strict. The local launcher remains backend-neutral and reports detected incompatibilities without selecting a different model or quantization.

Every request carries a stable request ID and records only safe routing metadata: selected mode, request provider, active provider, exact model, fallback reason, route generation, timing, and whether any tool effect occurred. Prompts, attachment contents, and credentials are excluded from logs.

## Legacy configuration migration

Legacy provider, model, endpoint, and mode values referring to GPT-OSS or Ollama are recognized only by the migration layer. Loading is side-effect free. The first explicit save writes the normalized schema atomically in the per-user configuration directory, retaining unrelated keys and user data. Repeated migrations yield the same result.

The process updates in-memory mode before persistence. If atomic persistence fails, the active session keeps the selected mode and exposes a recoverable warning; it never reports a false successful durable save.

## Runtime paths and logging

Relative mutable paths resolve below the operating system's per-user JARVIS directories. Explicit absolute paths remain supported for deployments and tests. Logs use owned handlers, atomic lifecycle management, redaction, and safe reinitialization after Qt or interpreter shutdown. A closed or externally detached stream does not produce an `EBADF` traceback.

## Latency and lifecycle

The desktop window is presented before optional heavy subsystems initialize. Health, voice diagnosis, indexing, and TTS are off the text-response critical path. Chat exposes time to first token separately from generation and total time, publishes progress, and supports cancellation.

The watchdog is progress-aware: activity renews its lease, partial model output is not replaced, and a recoverable typed timeout is reported. A fallback retry is allowed only before any emitted output or external tool effect. Tool-call and attachment identifiers make retries idempotent.

Speech uses a bounded single-consumer queue with replace/cancel semantics; completion and failure always release the slot. Text delivery does not wait for speech.

## Research and indexing

Research reports memory, retrieval, optional web, synthesis, and validation stages. Empty collections complete normally, sources are preserved, and partial results remain visible.

Indexing is persistent, incremental, pausable, cancellable, and scheduled below interactive chat. Startup scheduling does not block window creation.

## Attachments

The existing neutral attachment pipeline remains authoritative. The desktop `+` menu, multi-select, drop, and clipboard paths all call the same service and surface validation failures. Pending files survive mode changes and are removed only after a successful request. Real-format tests validate detection, processing, cancellation, retry, duplicate prevention, and provider routing.

## Verification strategy

Each defect starts with a failing regression test. Provider and endpoint behavior use deterministic fakes, including a local OpenAI-compatible HTTP server. Qt interaction uses `QTest`; file coverage uses small real temporary files. Live NVIDIA tests remain opt-in through `RUN_NVIDIA_LIVE_TESTS=true`. Final verification includes focused tests after each group, the full suite, Ruff, compileall, dependency checks, build/startup smoke tests, a visible Windows desktop smoke test, and scans for secrets, Base64 payloads, and active legacy provider names.

