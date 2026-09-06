# JARVIS Agentic LoRA B3.2 — runtime recovery

Date: 2026-08-31  
Status: **PARTIAL**

The frozen 600-case benchmark and decoding settings were preserved. The 12
valid rows were reused without rerun. The baseline runner now has a
`--manage-runtime` mode backed by the production `LocalModelRuntimeManager`:
it resolves `qwen35-9b-iq4-xs`, verifies model/profile/context/parallel,
reuses a compatible endpoint, recovers a transient owned-runtime loss with
the manager's bounded recovery budget, and cleans up only an owned process.

A zero-case managed-runtime smoke test passed and left port 8081 released.
The endpoint was not available for the remaining 588-case execution, so the
baseline remains incomplete and no metrics were fabricated.

## Current count

| Track | Completed | Required |
|---|---:|---:|
| model_surface | 37 | 380 |
| end_to_end | 0 | 220 |
| unseen_tool | 37 | 240 |
| hard_regression | 0 | 80 |
| total | 37 | 600 |

After one managed 25-case chunk, measured metrics are: tool selection 30/37
(81.08%), argument validity 37/37 (100.00%), non-empty final 37/37
(100.00%), runtime failures 0/37, and length finishes 0/37. Benchmark SHA-256 is unchanged:
`085989ef9480870f5c26e0fc72c0e34089d2bf9876109f884d7b8dfc87b9ef20`.

## User-runnable commands

From `C:\Users\shiki\Documents\jarvis\Jarvis`, run all remaining cases:

```powershell
Set-Location C:\Users\shiki\Documents\jarvis\Jarvis; $env:PYTHONPATH='src;.'; .venv\Scripts\python.exe -m training.agentic_lora.evaluation.run_baseline --manage-runtime --resume
```

Safe diagnostic chunk (25 missing cases):

```powershell
Set-Location C:\Users\shiki\Documents\jarvis\Jarvis; $env:PYTHONPATH='src;.'; .venv\Scripts\python.exe -m training.agentic_lora.evaluation.run_baseline --manage-runtime --resume --max-cases 25
```

No NVIDIA calls, cloud resources, training, full-BF16 download, production
code/model changes, or repository destructive operations occurred.

Validation: existing-result validator **PASS**; focused tests **9 passed**;
Ruff **PASS**; `git diff --check` **PASS**. B3 is not complete and B4
preflight is not authorized until 600/600 is present.
