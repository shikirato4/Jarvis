# JARVIS Backbone, Agent Runtime and Integral Stabilization Design

**Date:** 2026-08-23

**Status:** Approved for incremental implementation

**Repository constraint:** Work in the current dirty tree. Do not commit, push, reset, clean, destructively check out, overwrite `.env`, expose secrets, or revert pre-existing work.

## 1. Objective

Stabilize JARVIS end to end while preserving NVIDIA API, AUTO, future CUDA/NVFP4 support, and all existing public surfaces. Nemotron remains the conversational and agentic decision-maker. Deterministic runtimes become typed tools or internal implementation services instead of independent conversational responders.

Completion requires behavioral tests, real API/CLI/desktop execution, controlled NVIDIA requests when configured, LOCAL protocol tests, packaging, and an evidence-backed functional matrix. Passing `pytest` alone is insufficient.

## 2. Delivery order

The implementation is split into ten ordered, independently verifiable phases:

1. ConversationContextBuilder and real conversational context.
2. NemotronIdentity and LOCAL configuration without an NVFP4 hardcode.
3. AgentController and ActionExecutor.
4. Conversion of calculator, science, vision, data, web, files, code, Windows, image, and related conversational bypasses into agent tools.
5. RuntimeActivityStore and real runtime events.
6. Integral data analyzer.
7. Neural Router, dataset, training, baselines, benchmark, and shadow mode.
8. Activity Graph driven exclusively by RuntimeActivityStore.
9. Composer/UI, multimedia, image progress, TTS, vision, and remaining reproduced defects.
10. Exhaustive E2E audit and stabilization of every accessible function.

No phase advances while its directed tests, module suite, integration tests, and affected regressions have known correctable failures.

## 3. Current-state evidence

The initial tree contains 151 tracked modified files with 4,132 insertions and 3,763 deletions, plus untracked Nemotron and attachment work. These changes are treated as user-owned baseline state.

The following defects were reproduced before implementation:

- Desktop model requests contain only the system message and current user message. A sentinel introduced 20 turns earlier is absent from the final request.
- `calcula 781 × 94` is classified as `science` and performs zero model calls.
- The main composer is a fixed single-line `QLineEdit`; 501 pasted lines have a one-line size hint.
- LOCAL health accepts only an exact advertised NVFP4 identifier.
- Persistent configuration overwrites the local model with the NVFP4 constant on load/save migration.
- A sensitive-window capture can be represented as a degraded observation while verification still defaults missing success indicators to success.
- The repository has no Neural Router, training pipeline, shadow mode, runtime activity store, Activity Graph, or integral CSV/JSON/XLSX analyzer.
- A temporary backend startup measured about 7.94 seconds, with about 7.81 seconds in secondary runtimes.
- The baseline collects 727 tests. At least two attachment tests fail because `pypdf` is declared but absent from the active virtual environment; additional failures remain under collection.

## 4. Shared architectural invariants

### 4.1 Nemotron is the brain

Normal chat and agent mode follow this flow:

```text
User
  -> ConversationContextBuilder
  -> Nemotron
  -> AgentController
  -> typed tool proposal
  -> SecurityGate
  -> ActionExecutor
  -> structured observation
  -> ConversationContextBuilder
  -> Nemotron
  -> next tool or final answer
```

Fast paths remain only for internal protocol operations such as cancellation, confirmation replies, health probes, and UI bookkeeping. User-facing calculator, science, web, data, vision, files, code, image generation, and Windows requests do not produce final conversational answers before Nemotron sees their observations.

### 4.2 No chain of thought

The runtime stores and displays operational state, tool names, sanitized observations, timings, errors, confirmation state, and artifacts. It never requests, records, or displays hidden chain-of-thought.

### 4.3 Truthful state and progress

Terminal states are `completed`, `partial`, `blocked`, `failed`, or `cancelled`. A blocked or failed observation cannot be promoted to completed. Numeric progress is emitted only with a measured numerator and denominator. Otherwise the UI shows indeterminate activity.

### 4.4 Secrets and privacy

Events and traces contain identifiers, counts, classes, status, timing, and redacted diagnostics. They exclude raw prompts, attachment bytes, absolute private paths where not required, API keys, `.env` values, and unrestricted tool output.

## 5. ConversationContextBuilder

### 5.1 Responsibility

`ConversationContextBuilder` is the only component that assembles model messages for desktop chat and agent turns. It consumes immutable inputs and returns a `BuiltConversationContext` containing messages, token estimates, selection decisions, and privacy-safe telemetry.

Inputs are:

- system identity and current surface policy;
- conversation messages, including assistant messages;
- tool calls and tool observations paired by call ID;
- attachment-derived context and authorized native media references;
- retrieved persistent-memory candidates;
- the current user message;
- provider context limit and requested output budget.

### 5.2 Message model

Model messages gain structured support for `system`, `user`, `assistant`, and `tool`, including tool-call IDs and assistant tool-call metadata. Existing plain text messages remain compatible.

Conversation history and persistent memory remain separate sources. Tests use temporary databases and must prove that current-turn recall does not depend on persistent memory.

### 5.3 Budgeting and compaction

The builder reserves output tokens and safety margin before selecting input. It estimates tokens through an injectable estimator. Provider-specific tokenizers may replace the default conservative estimator without changing selection semantics.

Protected content includes:

- system policy;
- current user message;
- explicit user instructions to remember a value during the conversation;
- unresolved confirmations;
- tool-call/result pairs required for later steps;
- attachment context referenced by follow-up questions;
- the most recent coherent user/assistant turns.

Compaction preserves protected items verbatim, deduplicates repeated context by stable fingerprints, and summarizes only eligible older spans. If protected content cannot fit, the builder reports a structured overflow instead of silently dropping it. `context_compacted` and `context_truncated` are distinct events.

### 5.4 Required telemetry

Every build emits:

- `conversation_context_built`;
- `history_message_count`;
- `tool_observation_count`;
- `attachment_context_count`;
- `memory_candidate_count`;
- `memory_selected_count`;
- `estimated_context_tokens`;
- `context_compacted`;
- `context_truncated`.

Only counts, booleans, IDs, and timing are logged.

### 5.5 Acceptance

Automated and E2E tests introduce `COBALTO-721`, execute at least 20 intervening turns containing tool calls, attachment context, and a compaction boundary, then recover the sentinel from current-conversation context with persistent memory disabled.

## 6. NemotronIdentity and provider configuration

`NemotronIdentity` separates a canonical family identity from transport-specific advertised IDs.

Configuration contains:

- canonical family ID;
- requested inference model ID;
- exact accepted advertised aliases;
- backend kind: NVIDIA API, vLLM-compatible, LM Studio, llama.cpp, or generic OpenAI-compatible;
- configurable base URL;
- optional quantization/profile metadata used for diagnostics, not broad matching.

Validation normalizes casing and harmless path separators, then requires exact membership in the configured alias set. It does not accept arbitrary models containing `nemotron`. Default aliases cover the official remote ID and exact NVFP4 ID. GGUF or LM Studio names are accepted only when supplied explicitly through configuration.

Persistent configuration preserves the selected URL, model, aliases, and backend. Legacy GPT-OSS/Ollama variables are ignored for routing and reported by name only in a safe legacy-configuration diagnostic. `.env` is never rewritten.

AUTO preserves current coordinator semantics. NVIDIA and LOCAL remain strict. No hidden GPT-OSS fallback is introduced.

## 7. AgentController and ActionExecutor

### 7.1 AgentController

The controller owns one bounded agent run:

- build context;
- request Nemotron output with tool schemas;
- validate tool calls;
- execute approved calls;
- append observations;
- replan after recoverable failures;
- stop on a final answer, cancellation, deadline, repeated signature, or step limit.

The controller uses stable run, turn, step, and tool-call IDs. It deduplicates repeated effects and detects loops by normalized action signatures plus world-state fingerprints. Tool errors are observations, not synthetic assistant answers.

Retries occur only for declared transient failures and before a non-idempotent effect unless the tool supplies an idempotency key. Heartbeats and deadlines are measured with an injectable clock.

### 7.2 ActionExecutor

The executor receives typed `ActionRequest` objects and returns `ActionObservation` objects. Each action definition declares input schema, risk, timeout, idempotency, cancellation support, rollback support, and execution adapter.

Adapter preference is:

1. official API or SDK;
2. safe OS/subprocess interface;
3. filesystem API;
4. dedicated CLI;
5. Windows UI Automation;
6. vision-assisted interaction;
7. mouse/keyboard fallback.

The existing action registry, tool registry, system runtime, UI automation, code agent, and desktop-agent services are adapted rather than discarded. SecurityGate evaluates every proposed effect regardless of source. Confirmation approval is scoped to the exact action fingerprint and expires after use or material state change.

### 7.3 Calculator acceptance trace

The regression test captures this exact sequence:

```text
user_message
model_request
assistant_tool_call calculator.evaluate
security_allowed
tool_started
tool_succeeded
tool_result_message
second_model_request
assistant_final
```

Normal chat cannot call `science_runtime` directly for the calculation.

## 8. RuntimeActivityStore

The store subscribes to real runtime events and maintains a bounded, thread-safe projection of component nodes and flow edges.

Node states are `idle`, `queued`, `active`, `waiting`, `success`, `failed`, `blocked`, or `cancelled`. Transitions require an originating event with correlation and causation IDs. Edges are created only when one real component hands work or an observation to another.

The store supplies snapshots and subscriptions to API, desktop UI, tests, and Activity Graph. It never schedules work and cannot invent a timer-based success animation. Retention is bounded by event count and age.

## 9. Integral data analyzer

The analyzer is a typed agent tool with sandboxed inputs and outputs. It supports CSV, TSV, JSON records/tables, and XLSX without macros.

Operations include schema/profile, filtering, sorting, aggregation, arithmetic, descriptive statistics, anomaly detection, comparison, and plotting. Parsers handle encoding, delimiter inference, mixed columns, nulls, invalid dates/numbers, empty/corrupt files, and reasonable size limits. XLSX reads values only and rejects active content.

Results contain a compact observation, typed table metadata, warnings, provenance, and artifact IDs. Large tables are sampled or aggregated under an explicit budget. Generated outputs use the centralized user output directory, not the Git tree.

## 10. Neural Router

### 10.1 Role

The router produces candidate intent/tool/context rankings with scores. Nemotron or AgentController makes the final decision. The router cannot approve sensitive actions.

### 10.2 Model and data

The initial implementation uses deterministic hashed lexical features and a small NumPy multilayer classifier. The controlled dataset contains synthetic, reviewed examples for chat, calculator, web, data, image, vision, audio, video, code, files, Windows, Git, memory, agent, research, science, writing, Unity, and security. Private conversations and secrets are excluded.

The dataset is versioned and split deterministically into stratified train, validation, and test partitions. Training produces a candidate checkpoint plus metadata and label vocabulary.

Inference and training are accessed through backend-neutral contracts. `NeuralRouterBackend` owns candidate inference and checkpoint loading, while a separate trainer contract owns fitting and checkpoint production. Dataset schemas, feature records, benchmark reports, shadow comparisons, promotion policy, and AgentController depend only on those contracts and shared result models. NumPy is the first backend implementation, not a type or storage dependency of the surrounding system; a future PyTorch or other backend can replace it without changing routing, evaluation, promotion, or agent orchestration.

### 10.3 Baselines and promotion

Benchmarks compare:

- current deterministic rules;
- majority/random reference where meaningful;
- a simple linear classifier;
- the neural candidate.

Metrics include accuracy, macro F1, per-class precision/recall, top-1 tool accuracy, top-3 tool recall, latency, and checkpoint memory. The neural router remains disabled by default unless it clearly exceeds the relevant non-random baseline on held-out data and stays within configured latency/memory limits.

### 10.4 Shadow mode and traces

Shadow mode records candidate classes/scores and compares them to the action selected by Nemotron/controller without controlling execution. Optional traces contain intent class, candidate tool IDs, selected tool IDs, outcome, step count, latency, and correction-required flag. Raw prompts are not stored automatically. Training remains offline with explicit checkpoint promotion and rollback.

## 11. Activity Graph and UI

The graph is a desktop view backed by RuntimeActivityStore snapshots. Nodes represent only registered real components. Active edges correspond to correlated handoffs. Historical terminal states decay to idle based on explicit projection rules, not fabricated work.

The composer becomes a multiline editor that grows to a configured maximum height, then scrolls while preserving text, cursor, selection, paste, drag/drop, and attachment behavior. Enter sends and Shift+Enter inserts a newline, with accessible alternatives.

Message rendering must support long plain text, Markdown-like text, code, lists, Unicode, and unbroken URLs without fixed-height clipping. Conversation history remains scrollable during streaming and resize.

Image generation surfaces real states and measured step progress. Jobs without measurable steps show indeterminate progress. Completed artifacts show a real preview and safe open/reveal/copy actions.

## 12. Multimedia, voice, and vision stabilization

Attachment context remains request-scoped through streaming, retries, tool calls, synthesis, and follow-up turns. Video sampling uses duration-aware budgets and never sends unbounded frames. Audio/video dependency failures are explicit.

TTS provider capability is detected once per lifecycle. CPU fallback remains available; CUDA/NVFP4 paths remain intact and are reported as `BLOCKED_BY_HARDWARE` when not executable. Windows SAPI work must address COM apartment initialization, comtypes cache/type library mismatch, engine threading/lifecycle, and logging/file-descriptor closure rather than hiding the backend.

Sensitive-window capture remains blocked. A degraded or blocked observation propagates to verifier, controller, activity store, and final status without invented visual content.

## 13. Test strategy

Every behavioral change follows red-green-refactor:

1. add the smallest failing behavioral test;
2. run it and confirm the expected failure;
3. implement the minimal behavior;
4. rerun the directed test;
5. run the module suite;
6. run integration tests;
7. run affected regressions;
8. perform real E2E where applicable.

Temporary directories and databases isolate memory, files, Git, code-agent repos, automation, and outputs. External effects use fakes or safe temporary applications. NVIDIA live tests are few, controlled, secret-free, and conditional on configured credentials. GPU execution is never simulated.

Fault injection covers HTTP 401/403/404/429/500, timeout, disconnect, malformed/duplicate/empty chunks, provider death, tool exception, corrupt files, locked DB, cancellation, image interruption, TTS failure, and blocked vision.

## 14. Final verification and reporting

The final gate runs the complete test suite, Ruff check and format check, `git diff --check`, compileall, `pip check`, wheel and sdist builds, isolated wheel installation/import smoke, real API, real CLI, `python -m jarvis.desktop`, E2E scenarios, shutdown, and a second complete regression run after late shared-code fixes.

The report inventories every public API route, CLI command, desktop control, registered action/tool/capability, and documented function with evidence and one of: `PASS`, `FAIL`, `PARTIAL`, `BLOCKED_BY_HARDWARE`, `BLOCKED_BY_DEPENDENCY`, or `NOT_APPLICABLE`.

Git final state explicitly confirms no commit, push, reset, destructive checkout, clean, `.env` overwrite, secret exposure, or pre-existing work reversal.
