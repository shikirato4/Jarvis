# JARVIS Agentic LoRA — B3 frozen Qwen3.5 BASE baseline

Status: **PASS** · frozen 2026-09-01 (offline closeout; zero inference)

The persisted evidence contains exactly 600/600 unique benchmark IDs, with no
duplicates, malformed rows, missing scoring fields, raw CoT, or
`reasoning_content`. The benchmark is unchanged and the accepted result rows
were not rewritten.

## Frozen identity

| Field | Value |
|---|---|
| Model/profile | Qwen3.5-9B-IQ4_XS / qwen35-9b-iq4-xs |
| GGUF SHA-256 | `7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429` |
| Benchmark SHA-256 | `085989ef9480870f5c26e0fc72c0e34089d2bf9876109f884d7b8dfc87b9ef20` |
| Results SHA-256 | `8468dc0d5c0e3ab9cc8b737a6e0dd9cce8cc523fb0c00911f46ecb742f9795ec` |
| Context/parallel | 8192 / 1 |
| Thinking/temp/top-p/max tokens | OFF / 0 / 1 / 768 |
| llama.cpp/provider | b10516-b95502ba9 / qwen_local |
| Scoring | agentic-eval-v1-score-v1 |

## Coverage and metrics

- Model-surface: **380/380**; end-to-end: **220/220**.
- Tool selection: **386/600 (64.33%)**.
- Argument validity: **600/600 (100.00%)**; argument-accuracy proxy: **600/600 (100.00%)** overall. Category-level argument records are separately retained where available.
- Action-vs-information: **5/12 (41.67%)**; capability-present: **5/12 (41.67%)**.
- Multi-tool: **8/11 (72.73%)**; missing-required-tool category: **10/21 (47.62%)**.
- Unseen-tool selection/success: **174/240 (72.50%)**.
- Non-empty final: **600/600 (100.00%)**; `finish_reason=length`: **0/600**; invalid structured output: **0/600**; tool loops: **0/600**.
- Spanish: **340 cases**, selection **226/340 (66.47%)**, non-empty **340/340**. English: **260 cases**, selection **160/260 (61.54%)**, non-empty **260/260**. Mixed: unavailable (0 cases).
- Runtime failures: **0/600**. Model decision failures are reflected in metric misses and are not counted as runtime failures.

Observation grounding, security allow/confirmation/denial, false-success,
false-capability-denial, and continuation semantics are represented by their
benchmark categories, but the current persisted scoring surface does not emit
independent truth labels for those dimensions; they are therefore recorded as
**not independently measurable** rather than inferred from tool-selection.

Failure-category counts: unseen_tool 240; output_completion 61; preservation
50; false_success 21; missing_required_tool 21; observation_grounding 21;
scope_honesty 21; security_confirmation 21; security_denied 21; tool_failure
21; tool_loop 21; false_capability_denial 11; action_vs_information 12;
argument_accuracy 12; capability_present 12; tool_selection 12;
non_empty_final 11.

Timing: total **2,509,396.65 ms**, average **4,182.33 ms**, median
**3,766.18 ms**, p90 **7,106.49 ms**, p95 unavailable, slowest **35,513.65
ms**. Model-surface average **4,472.42 ms**; end-to-end average **3,681.25
ms**.

## Freeze and B4 contract

Freeze manifest: [base_freeze_manifest.json](C:/Users/shiki/Documents/jarvis/Jarvis/training/agentic_lora/data/evaluation/base_freeze_manifest.json)

Future BASE-vs-LoRA comparisons must keep the benchmark, schemas, SecurityGate,
runtime surface, context, thinking, temperature, top-p, explicit output cap,
and scoring constant. B4 must perform full BF16 text-backbone, target-module,
PEFT injection/save/reload, and forward/generation smoke checks before any
optimizer step. No B4 work was started here.

Validation: focused tests **9 passed**; Ruff **PASS**; `git diff --check`
**PASS**. No production code/model changes, training, cloud, NVIDIA calls, or
model requests occurred in B3.3.
