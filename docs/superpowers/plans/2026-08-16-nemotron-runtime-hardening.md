# JARVIS Nemotron Runtime Hardening Implementation Plan

> Execute in the current worktree. Preserve all pre-existing changes. Do not commit, push, reset, clean, or mass-format.

**Goal:** Remove active GPT-OSS/Ollama behavior and eliminate the reported provider, persistence, logging, voice, latency, research, indexing, benchmark, and attachment regressions while retaining the approved Nemotron hybrid architecture.

**Architecture:** Keep the existing provider coordinator, normalized model contracts, attachment pipeline, desktop runtime, operation registry, and capability boundaries. Harden them with strict invariants, user-scoped storage, idempotent migration, background scheduling, and typed diagnostics.

**Stack:** Python 3.11+, Pydantic, httpx, FastAPI, PySide6, pytest, Ruff, platformdirs.

---

### Task 1: Freeze baseline and add migration/provider regression tests

- Add tests proving only the three approved modes route requests.
- Add tests reproducing the incorrect legacy error message.
- Add idempotent migration tests for legacy configuration without active legacy providers.
- Run the focused tests and confirm they fail for the intended reason.
- Remove legacy registrations, ModelService branches, defaults, diagnostics, UI/CLI/scripts, and active documentation references.

### Task 2: User-scoped persistence and logging lifecycle

- Add tests for side-effect-free reads, atomic saves, repeated changes, restart persistence, migration, failure reporting, redaction, closed streams, rotation, and reinitialization.
- Introduce platform-specific data/config/log paths while preserving explicit absolute overrides.
- Make mode changes immediate in memory and durable through atomic replacement.
- Harden owned logging handlers and eliminate repository log writes.

### Task 3: Lightweight voice status and bounded asynchronous TTS

- Prove normal status refresh and mode changes do not import or initialize `pyttsx3`/COM.
- Add queue saturation, cancellation, replacement, and lock-release tests.
- Cache lightweight status; reserve backend initialization for explicit diagnosis or synthesis.
- Move speech outside chat completion and enforce a bounded worker queue.

### Task 4: Streaming, progress-aware watchdog, and request identity

- Add TTFT/progress/cancellation/partial-output/no-duplicate-tool regression tests.
- Replace legacy error translation with provider-neutral typed diagnostics.
- Renew watchdog leases on progress and prevent fallback after output/effects.
- Emit safe request/provider/timing metadata and keep health checks off the request path.

### Task 5: Attachment UI and unified pipeline end to end

- Add QTest coverage for menu, keyboard, multi-select, drop, paste, chips, remove, cancel, retry, and preservation across mode changes.
- Add real temporary PNG/JPEG/text/Markdown/JSON/CSV/PDF/DOCX/WAV/MP4 fixtures.
- Surface service validation errors in the UI and verify deduplication and successful cleanup semantics.
- Verify strict LOCAL creates no NVIDIA traffic.

### Task 6: Research, indexing, startup, and benchmark truthfulness

- Add tests for empty retrieval collections, staged progress, sources, and partial research.
- Make indexing scheduling non-blocking and lower priority than chat; verify pause/cancel/incremental persistence.
- Defer optional desktop startup work until after the window is visible.
- Partition benchmark measurements and label simulated/local-unavailable results without fabricated throughput.

### Task 7: Documentation and complete verification

- Update README, master context, and architecture/capability documents; retain historical records unchanged.
- Compare every functional diff with README coverage.
- Run focused and full tests, Ruff, compileall, `pip check`, build, API/backend/frontend startup and smoke checks.
- Run visible Windows desktop interaction when supported.
- Scan tracked/untracked project text for credentials, large Base64 payloads, and active `gpt-oss`, `gpt_oss`, or `ollama` use; classify any historical-only matches.
- Report exact evidence and external/hardware blockers without claiming unverified NVFP4 compatibility.

