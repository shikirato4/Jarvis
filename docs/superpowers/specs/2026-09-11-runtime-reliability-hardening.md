# P3.3 runtime reliability hardening

## Result

**P3.3: PARTIAL.** The deterministic implementation, regression suite, and
machine-readable diagnostics pass. The mandatory post-change live-local matrix
could not be completed because the configured CUDA `llama-server.exe` exits
before model readiness with process code `3236495362`, and no Vulkan fallback
binary is installed. This is a runtime-launch infrastructure failure, not a
research, prompt, context, or completion-policy failure. Phase 4 is therefore
not ready until the live matrix is rerun successfully.

No training was started, no cloud model was used, and cost was **$0**.

## Scope and invariants

This phase hardens only research state, stopping, citations, synthesis repair,
and user-visible output completion. It does not redesign PRE-LORA-2. It does
not modify provider selection, Brave Search configuration, keys, or `.env`
values. It does not change the production GGUF, local context size, B4A, or the
frozen B3 corpus and reports.

The P3.3 changes owned by this pass are limited to:

- `src/jarvis/agent_runtime/controller.py`
- `src/jarvis/models_runtime/output_completion.py`
- `src/jarvis/desktop_runtime/chat.py`
- `tests/test_agent_runtime.py`
- `tests/test_chat_engine.py`
- `training/agentic_lora/evaluation/run_p33_diagnostics.py`
- `training/agentic_lora/data/p33_diagnostics.json`
- this specification

The worktree contains other pre-existing and concurrent changes. They were
preserved and were not reset, committed, or attributed to this phase.

## End-to-end research ownership

The research flow has one bounded owner for each responsibility:

1. `DesktopChatEngine` parses `/deepresearch` and dispatches the request.
2. `WebResearchOrchestrator` owns intent decomposition, query planning,
   provider search, fetch, normalization, ranking, evidence extraction, and
   stopping.
3. `WebResearchSession` is canonical mutable state: question, requirements,
   queries, candidates, opened sources, evidence, citation records, metrics,
   budget state, and stop reason.
4. Source identity is derived from the normalized URL. Display citations are
   assigned from final evidence as stable `S#` labels.
5. `build_synthesis_prompt` packs bounded, machine-readable `[S#]` evidence
   blocks with title, location, evidence, requirement coverage, and an explicit
   untrusted-data boundary.
6. `RuntimeService -> ModelService -> OutputCompletionPolicy` produces one
   completed logical synthesis before citation validation.
7. `DesktopChatEngine` normalizes citation syntax, removes generated reference
   sections, validates IDs and claims, checks requirement coverage and minimum
   report length, and performs at most three total synthesis attempts.
8. Only validated IDs are rendered as links. Unknown numeric citations are not
   promoted to source IDs. Invalid or incomplete synthesis returns an explicit
   bounded fallback with a diagnostic reason.

Research stopping is based on required coverage, contradiction state,
information gain, and hard query/page/round limits. Query deduplication and
stagnant-round detection prevent unbounded search loops.

## Root-cause analysis

### Deep Research source-only fallback

The legacy synthesis request set an explicit `max_tokens` cap. An explicit,
non-adaptive cap is authoritative in `OutputCompletionPolicy` and therefore
sets `disable_auto_continuation`. When the local model ended with
`finish_reason=length`, citation validation received an incomplete draft and
the chat layer could fall back to sources without a completed synthesis.

The first failing layer was synthesis completion configuration, before citation
parsing or rendering. The hardened request uses an adaptive budget, an explicit
research output ceiling, and at most two textual continuations. Citation
validation and bounded repair run only after that logical response completes.

Provider failures now record a stable `synthesis_error` and a
`fallback_reason` prefixed with `synthesis_provider_failure:`.

### Simple Files information truncation

The agent controller labelled any first text response as `agent_initial`, while
the completion policy explicitly excluded `agent_initial` from continuation.
Consequently an informational Files answer could stop at `length` even though
the desktop request opted into auto-continuation.

`agent_initial` text is now eligible for bounded textual continuation. The
`structured_tool_call` phase remains ineligible, and continuation requests
remove both `tools` and `tool_choice`. Tool-call JSON is never repaired by text
continuation; the controller retains its separate single structured retry.

### PRE-LORA-2 research exact-sequence score

The historical 50% exact-sequence score mixed three different outcomes:

| Arm | Exact sequence | Task success | Missing required search | Extra search |
|---|---:|---:|---:|---:|
| control | 50% | 100% | 1/2 | 0/2 |
| compact | 50% | 100% | 1/2 | 0/2 |
| full | 50% | 50% | 0/2 | 1/2 |

In control and compact, one case completed successfully with one sufficient
search although the fixture expected two calls. In full, one case made a
bounded extra search and failed to produce a final answer. Exact call count is
therefore retained as an efficiency signal, not treated as task correctness.
Dedicated research diagnostics now report task success, source sufficiency,
duplicate queries, missing searches, extra searches, coverage, and fallback
state separately.

## Global output-completion matrix

All user-visible model text converges through
`RuntimeService -> ModelService -> OutputCompletionPolicy`. Caller-provided
hard caps remain authoritative and disable automatic continuation unless the
caller explicitly opts into adaptive completion.

| Path | Initial policy at 8,192 context | Continuation and stopping owner |
|---|---|---|
| Normal chat | target up to 768; minimum 384 | `ModelService`; max 2 continuations; logical ceiling 35% of context |
| Short chat | target 160; minimum 96 | `ModelService`; max 2; total at most 20% of context |
| Long chat | target up to 2,048; minimum 768 | `ModelService`; max 2; total at most 45% of context |
| Extended chat | target up to 4,096; minimum 2,048 | `ModelService`; max 8; rolling public state; 24,000 visible-token hard ceiling |
| Agent initial text | target up to 1,024; minimum 768; reserves 1,024 follow-up tokens | `ModelService`; max 2; text only; tools removed from continuation |
| Structured tool call | target up to 1,024; minimum 512 | `AgentController`; no textual continuation; one structured retry |
| Post-tool final | target up to 1,536; minimum 768 | `ModelService`; max 2; total at most 45%; no tools in continuation |
| Multi-tool continuation | same phase-specific agent rules | `AgentController` controls the loop; only final public text can auto-continue |
| Files information | agent-initial information rules | `ModelService`; `length` now continues as one logical answer |
| Research planning/search | no LLM text continuation | `WebResearchOrchestrator`; coverage/gain/query/page/round bounds |
| Deep Research synthesis | adaptive long-form, ceiling 1,800; normal research ceiling 1,200 | `ModelService`; max 2 before validation |
| Citation repair | same evidence prompt and verified IDs | `DesktopChatEngine`; max 3 attempts total; simplified retry capped at 1,200 |
| Desktop streaming | phase-specific adaptive budget | `ModelService`; one visible logical stream across segments |
| Non-stream runtime service | phase-specific adaptive budget | `ModelService`; one merged logical response |

Continuation requires a normally closed stream, `finish_reason=length`, no tool
calls, explicit `auto_continue_text`, remaining context/response budget, and an
unused continuation slot. It stops on `stop`, cancellation, provider failure,
context exhaustion, total-response limit, repeated overlap/loop guard, or the
configured continuation maximum. Rolling extended state contains only public
request/answer text; it does not expose hidden reasoning.

## Deterministic diagnostics

`training/agentic_lora/evaluation/run_p33_diagnostics.py` is network-free and
does not invoke Brave. Its persisted JSON result is
`training/agentic_lora/data/p33_diagnostics.json`.

The run passed with:

- completed research and `stop_reason=evidence_sufficient`;
- one round, four planned searches, zero duplicate queries, and two sources;
- stable source hashes and stable `S1`/`S2` synthesis prompts;
- valid parsed citations, verified rendered links, and no promotion of unknown
  numeric citation `[99]`;
- a forced `length -> stop` model sequence merged into one logical answer;
- one continuation, tools removed from that continuation, and final reason
  `stop`;
- diagnostic flags confirming no Brave modification, no cloud use, no training,
  and $0 cost.

## Regression coverage

The final focused suite passed **239 tests**. It covers output budgets,
completion overlap and loop guards, cancellation, context/total ceilings,
streaming and non-streaming merging, structured tool-call isolation, agent
initial information continuation, post-tool finals, research intent and
freshness, ranking/deduplication/diversity, bounded retries, offline behavior,
SSRF and redirect checks, contradiction handling, PDF evidence, context
packing, requirement coverage, fabricated citation removal, verified links,
provider retry, citation repair, and fallback after the third invalid attempt.

New direct regressions verify:

- `agent_initial` information text completes across `length -> stop` without
  forwarding tool schemas; and
- Deep Research preserves `[S1]` in both initial and continuation requests and
  renders the validated citation in the final response.

## PRE-LORA-2 and frozen B3

No schema-count optimization was made because the truncation root cause was the
completion-phase exclusion, not schema pressure. The PRE-LORA-2 production
token measurements remain:

| Component | Tokens |
|---|---:|
| Full static prompt | 3,169 |
| 96-schema family manifest | 271 |
| Selected 12-schema index | 136 |
| Selected 12 exact schemas | 2,348 |
| Representative rendered Files request | 6,177 |
| Margin before generation in 8,192 | 2,015 |
| Margin after 1,024 output reserve | 991 |

The frozen B3 manifest, canonical and compiled datasets, baseline artifacts,
and recorded hashes remain unchanged. No B3 evaluation input was reused for
P3.3 development.

## Live-smoke status

| Smoke | Result | Evidence / blocker |
|---|---|---|
| Earlier real local Qwen + AgentController Files action | PASS | Persisted in `training/agentic_lora/data/prelora2_files_smoke.json` before this P3.3 change |
| Deterministic Deep Research | PASS | Stable sources, synthesis prompt, citations, and bounded stopping |
| Deterministic forced long response | PASS | `length -> stop`, one logical answer, one continuation |
| Post-change simple informational prompt | BLOCKED | local CUDA `llama-server.exe` exits before readiness |
| Post-change Files informational prompt | BLOCKED | same model-server launch failure |
| Post-change forced long local response | BLOCKED | same model-server launch failure |
| External non-Brave research smoke | SKIPPED | no configured non-Brave search provider was available |

The host reports an NVIDIA RTX 5070 Laptop GPU with available memory and no
stale `llama-server` process. Direct server startup still exits before emitting
a usable readiness response, while the Vulkan fallback executable is absent.
This phase deliberately does not replace binaries, switch GGUFs, reduce context,
or alter provider configuration to work around that environmental failure.

## Known limits and next gate

- Citation support is lexical/evidence-bound validation, not formal entailment.
- Provider/network quality can still exhaust bounded research budgets.
- A provider crash cannot be repaired by the completion layer; it produces an
  explicit diagnostic fallback.
- The live post-change simple/Files/long matrix remains mandatory.
- External research should be rerun only when a configured non-Brave provider
  is available, preserving the explicit provider boundary for this phase.

**READY FOR PHASE 4: NO.** Repair or restore the existing local model-server
runtime outside P3.3 scope, then rerun the three post-change local smokes and a
non-Brave external Deep Research smoke. Promote P3.3 to PASS only if those
complete without truncation, source-only fallback, invalid citations, or an
unbounded loop.
