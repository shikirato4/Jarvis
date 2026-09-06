# JARVIS Agentic LoRA B3.1 — frozen base baseline

Date: 2026-08-31  
Status: **PARTIAL**

## Frozen execution result

The 600-case benchmark and scoring were not changed. The pre-existing 12 valid
rows were validated and retained; the resumable runner now supports
`--resume`, `--max-cases`, `--track`, and `--start-after`, persisting each row
immediately. A three-case endpoint probe encountered a server reset/refusal;
those runtime-incomplete rows were discarded and did not count as completions.
The user-managed production server was not restarted or modified.

| Track | Required | Completed |
|---|---:|---:|
| model_surface | 380 | 12 |
| end_to_end | 220 | 0 |
| unseen_tool (subset) | 240 | 12 |
| hard_regression (subset) | 80 | 0 |
| **total** | **600** | **12** |

Measured rows: tool selection 9/12 (75.00%); argument validity 12/12
(100.00%); non-empty final 12/12 (100.00%); runtime failures 0/12; length
finishes 0/12; invalid structured output 0/12. Spanish subset: 6/8 selection
(75.00%), 8/8 non-empty (100.00%). Durations: total 90,102.60 ms, average
7,508.55 ms, median 4,847.64 ms, p90 13,106.64 ms, slowest 35,513.65 ms;
model-surface average 7,508.55 ms; end-to-end not measured.

Benchmark SHA-256: `085989ef9480870f5c26e0fc72c0e34089d2bf9876109f884d7b8dfc87b9ef20`  
Results SHA-256: `437ba843b10d1cc674c58c7a8cada8f49ba1dec1976776d98e58b2b635687656`  
GGUF SHA-256: `7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429`  
Profile/model: `qwen35-9b-iq4-xs` / `Qwen3.5-9B-IQ4_XS`  
Runtime: `qwen_local`, context 8192, parallel 1, thinking off, temperature 0,
top-p 1, max tokens 768, scoring `agentic-eval-v1-score-v1`.

## Safety and readiness

- Production code modified: **NO** (evaluation tooling and generated evidence only).
- Production model/GGUF modified: **NO**.
- Training started: **NO**; cloud provisioned: **NO**; large model downloaded: **NO**.
- Raw chain-of-thought/private reasoning: **NO**; privacy contract: **PASS**.
- Output-length regression metric: **defined** and retained (`finish_reason=length`).
- Restart-safe: **YES**; frozen benchmark hash: **YES**.
- B3 now complete: **NO** (600-case baseline is incomplete).
- Ready for B4 preflight: **NO**; blocker is completion of all 600 cases through a live compatible endpoint.

## Artifacts

- Runner: `training/agentic_lora/evaluation/run_baseline.py`
- Existing-row validator: `training/agentic_lora/evaluation/validate_results.py`
- Results: `training/agentic_lora/data/evaluation/baseline_results.jsonl`
- Summary: `training/agentic_lora/data/evaluation/baseline_summary.json`
- Frozen benchmark/manifest: `training/agentic_lora/data/evaluation/frozen_benchmark.jsonl` and `benchmark_manifest.json`
- Reproducibility manifest: `training/agentic_lora/data/b3_manifest.json`

## Validation

`9 passed` focused tests; Ruff: **PASS**; `git diff --check`: **PASS** (only
pre-existing line-ending warnings). No commit, push, reset, checkout, or
destructive cleanup was performed.

The next authorized action is B3.1 resumption only after the runtime endpoint
is available; stop here and do not begin B4 automatically.
