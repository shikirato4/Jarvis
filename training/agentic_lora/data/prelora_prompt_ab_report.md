# PRE-LORA-1 prompt A/B report

## Result

**PARTIAL.** The scaffold is integrated, bounded, removable, schema-driven, and
passes focused runtime tests. The seeded prompt-only A/B is healthy but does not
show an aggregate tool-selection improvement, so this report does not claim one.

## Evaluation protocol

- Model/profile: `Qwen3.5-9B-IQ4_XS` / `qwen35-9b-iq4-xs`.
- Context: 8,192; parallel: 1; local managed runtime only.
- Source: canonical validation split, never frozen B3 held-out cases.
- Sample: 23 cases per arm (46 requests): 11 current real-tool-shaped, 10
  fictitious/unseen-tool, and 2 non-tool control cases across 12 categories.
- Determinism: temperature 0, top-p 1, seed 42, max output 512.
- Control: compact current-production-style action/observation policy.
- Candidate: permanent core + pre-LoRA scaffold + dynamic capability metadata.

## Metrics

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| Tool-selection accuracy | 73.91% | 73.91% | 0.00 pp |
| Argument validity | 100.00% | 100.00% | 0.00 pp |
| Unseen/fictitious-tool selection | 80.00% | 80.00% | 0.00 pp |
| Real-tool selection | 63.64% | 63.64% | 0.00 pp |
| Capability-absent honesty | 100.00% | 100.00% | 0.00 pp |
| Action-vs-information | 50.00% | 50.00% | 0.00 pp |
| Unnecessary-tool rate | 4.35% | 4.35% | 0.00 pp |
| Nonempty first response | 100.00% | 100.00% | 0.00 pp |
| Mean prompt tokens | 347.61 | 884.13 | +536.52 |
| Mean latency | 5,024.51 ms | 6,948.43 ms | +1,923.92 ms |

Category movement was mixed: initial tool selection for `tool_failure` improved
from 0/2 to 1/2, while `no_false_capability_denial` fell from 2/2 to 1/2. Both
both arms passed one of two action-vs-information examples. With two cases per category
at most, these slices are diagnostic and not statistically strong.

The persisted machine-readable evidence is
`prelora_prompt_ab_results.jsonl` and `prelora_prompt_ab_summary.json` beside this
report. The runner is `training.agentic_lora.evaluation.run_prompt_ab`.

## What this A/B does and does not measure

It directly measures first-response tool choice, schema-valid arguments, unseen
schema use, action-vs-information, capability-present/absent behavior, unnecessary
calls, response presence, prompt tokens, and latency. The selected validation
records include multi-tool and failure categories, but this initial-decision runner
does not execute synthetic tools or score full recovery trajectories. Therefore it
does not claim model-level false-success or loop improvements. Existing controller
tests independently verify bounded loops, failure replanning, observation-grounded
terminal status, nonempty finals, SecurityGate denial, streaming, cancellation,
and continuation behavior.

## Frozen evidence

- Frozen benchmark unchanged: **YES**.
- Frozen baseline results/summary/manifest unchanged: **YES**.
- Compiled dataset unchanged: **YES**.
- Training: **NOT STARTED**.
- Cloud cost: **$0**.

## LoRA targets exposed by this baseline

- Reliably distinguish “how can I” information from an actual request to act.
- Prefer a named available capability without prompt sensitivity or omission.
- Preserve no-false-capability-denial behavior while recovering from failure.
- Learn multi-step continuation and verification from complete trajectories.
- Recover from typed failures without identical retries or premature prose.
- Produce truthful, grounded finals consistently with less static instruction.

The scaffold is suitable to keep as a temporary, feature-gated baseline, but
PRE-LORA-1 remains PARTIAL until a broader dev evaluation demonstrates a positive
behavioral delta (or an explicitly accepted neutral trade-off) and full trajectory
metrics are added.
