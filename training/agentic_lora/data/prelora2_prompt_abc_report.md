# PRE-LORA-2 prompt A/B/C report

## Result

**PASS as temporary production scaffolding.** Compared with the off control, Full
raises exact trajectory/tool selection from 85.71% to 92.86%, raises the combined
action-versus-information slice from 66.67% to 100%, and reduces unnecessary tool
calls from 7.14% to 0%. Argument validity, failure recovery, capability-absent
honesty, capability-present recognition, and unseen-schema use remain 100%.

This is a small deterministic development set, not a claim of statistical
generalization. The prompt remains temporary; the latency and context cost still
justify Agentic LoRA.

## Protocol

- Model: `Qwen3.5-9B-IQ4_XS`, local only.
- Runtime: exact `qwen35-9b-iq4-xs`, context 8,192, parallel 1.
- Arms: `off` control, PRE-LORA-1 `compact`, PRE-LORA-2 `full`.
- Set: 14 newly authored development cases per arm; frozen B3 was not used.
- Determinism: temperature 0, top-p 1, seed 42, max output 512.
- Evaluation: complete synthetic tool-observation trajectories, including
  two-step Files flows and a primary-to-backup recovery.
- TTFT: unavailable because the evaluator deliberately used non-streaming calls.

## Aggregate metrics

| Metric | Control | Compact | Full |
|---|---:|---:|---:|
| Exact tool/trajectory selection | 85.71% | 92.86% | 92.86% |
| First-tool accuracy | 92.86% | 100.00% | 100.00% |
| Argument validity | 100.00% | 100.00% | 100.00% |
| Action vs information | 66.67% | 100.00% | 100.00% |
| Capability present | 100.00% | 100.00% | 100.00% |
| Capability absent | 100.00% | 100.00% | 100.00% |
| Unseen tool | 100.00% | 100.00% | 100.00% |
| Multi-tool continuation | 92.86% | 92.86% | 92.86% |
| Failure recovery | 100.00% | 100.00% | 100.00% |
| False-execution rate | 0.00% | 0.00% | 0.00% |
| Unnecessary-tool rate | 7.14% | 0.00% | 0.00% |
| Nonempty-final rate | 92.86% | 100.00% | 92.86% |
| Mean cumulative prompt tokens | 1,306.79 | 2,071.50 | 7,417.14 |
| Mean total latency | 4,487.24 ms | 8,498.94 ms | 6,184.75 ms |

Research exact-sequence accuracy is 50% in every arm. The remaining miss is a
bounded extra/missing search in the synthetic sequence, not a capability denial or
false success. Full's nonempty-final miss likewise remains a LoRA target.

## Real local Files smoke

The smoke runner created only `PRELORA2_SAFE_FIXTURE.txt` inside an isolated
temporary workspace, then used the real local Qwen, AgentController, SecurityGate,
ActionExecutor, and FileNavigator path.

- Full request manifest contained `files: AVAILABLE_VIA_TOOL`.
- `files.search` survived progressive disclosure among 12 selected schemas.
- SecurityGate classified it `safe` and returned `allowed`.
- ActionExecutor returned a successful, verified observation with the fixture's
  exact path and metadata.
- The final answer repeated that exact observed path and did not deny file access.
- The information control produced an explanatory answer with zero observations
  and no tool execution. Its single model response ended at the output limit; this
  is recorded honestly and did not cause an action or false execution claim.

Machine-readable evidence:

- `prelora2_prompt_abc_results.jsonl`
- `prelora2_prompt_abc_summary.json`
- `prelora2_files_smoke.json`

## Architecture and regression evidence

One semantic owner now selects tools: AgenticPromptPolicy. QwenLocalProvider keeps
the selected count and order and performs only wire serialization. The dynamic
manifest uses all 96 current executable schemas while only 12 detailed schemas are
sent for the representative Files request.

Focused tests cover policy profiles and token ceilings, the exact 200-name
universe, dynamic family visibility, unfamiliar schemas, agent loops, calculator,
SecurityGate/ActionExecutor outcomes, activity graph/store, provider preservation,
conversation context, output continuation, and NVIDIA provider abstraction.
Streaming, cancellation, memory/context ordering, and hidden-thought boundaries
remain on their existing paths; no bypass was added.

## Frozen and cost controls

- Frozen B3 benchmark/evidence: unchanged.
- Compiled training dataset: unchanged.
- Training: **NOT STARTED**.
- Cloud: **NOT USED**.
- Cost: **$0**.

## What Full improves

Full gives a 9B model an explicit system identity, a complete but status-neutral
capability map, request-time awareness of every currently executable family, and
clear action-versus-information and discovery protocols. It fixes the architectural
case where provider-side semantic compaction could erase a Files tool already
selected by the agent policy.

## What still requires LoRA

LoRA is still needed to internalize this behavior with far fewer prompt tokens,
lower latency, more reliable final completion, stronger research sequencing, and
more consistent multi-step behavior across a much larger held-out distribution.

**Ready to use PRE-LORA-2 as temporary production scaffolding: YES.**
