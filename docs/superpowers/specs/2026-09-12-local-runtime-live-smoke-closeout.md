# P3.3R local runtime and live-smoke closeout

## Result

**P3.3R: PASS.** The pinned production CUDA runtime is ready again, the three
mandatory post-change local smokes pass, deterministic Deep Research remains
green, and one real external Deep Research run passed using only configured
non-Brave providers. P3.3 may be promoted from PARTIAL to PASS.

No production source file, runtime binary, DLL, model, context setting, provider
configuration, PRE-LORA-2 behavior, or frozen training artifact was changed.
This closeout adds only reproducible evaluation runners, their JSON results, and
this report.

## Local runtime root cause

The first failing layer in the historical `0xC0E90002` launches was the Windows
user-mode Code Integrity loader decision, while `llama-server.exe` attempted to
load adjacent `llama.dll`. The process did not reach the llama.cpp banner,
argument parsing, CUDA initialization, model open, GGUF metadata, tensor
allocation, HTTP startup, or the JARVIS readiness probe.

| Field | Evidence |
|---|---|
| Unsigned exit code | `3236495362` |
| Signed representation | `-1058471934` |
| Hex | `0xC0E90002` |
| Verified symbolic name | `STATUS_SYSTEM_INTEGRITY_POLICY_VIOLATION` |
| Faulting component | `runtimes/llama-cpp-cuda/llama.dll` |
| Windows events | Code Integrity Operational 3033 and 3077 |
| Policy ID | `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` |

`certutil -error 0xC0E90002` supplied the symbolic mapping. The Code Integrity
events state that `llama-server.exe` attempted to load `llama.dll`, which did
not meet Enterprise signing requirements or violated the active policy.
`Win32_DeviceGuard` reports both system and user-mode Code Integrity enforcement
enabled. `llama.dll` and `llama-server.exe` are unsigned.

The last matching rejection events occurred at 01:11:26 local time on
2026-09-12. The existing JARVIS log already contained successful launches later
that morning, before P3.3R diagnosis began. File timestamps and hashes did not
change. The precise external policy/reputation state transition that changed the
decision is not recorded by an available event, so it is classified as
**unknown** rather than attributed to a guessed Defender, Windows Update, or
driver action.

Because the exact pinned files were already loadable and healthy, the smallest
evidence-backed repair was **no artifact mutation**. Replacing or re-signing the
working build, changing the policy, or installing a different backend would
have been unjustified and higher risk.

## Binary set

| Item | Value |
|---|---|
| Server | `runtimes/llama-cpp-cuda/llama-server.exe` (ruta relativa al repo) |
| Size | 9,216 bytes |
| Modified UTC | 2026-08-20 17:43:28 |
| SHA256 | `5A3CBD5613C45EF2D53D3AFC6734FD9E67229C0066C2415626DDC7C18901D36C` |
| Build | `10516` (`b10516`) |
| Commit | `b95502ba9` |
| Compiler/architecture | Clang 20.1.8, Windows x86_64 |
| `llama.dll` SHA256 | `FDABE60BDB0673411688572F752939DE80D50D998331E4EFE94FE28F24FC504D` |

The folder contains the matching split executable/implementation layout,
including `llama-server-impl.dll`, `llama-common.dll`, `llama.dll`, `ggml.dll`,
`ggml-base.dll`, CPU backends, `ggml-cuda.dll`, `cudart64_13.dll`,
`cublas64_13.dll`, and `cublasLt64_13.dll`. There is no Mark-of-the-Web stream.

Coherence was demonstrated operationally rather than inferred from timestamps:
the server and CLI report the same pinned build, sibling binaries load the
adjacent CUDA/RPC/CPU backends, the server loads the production model, and
generation completes. The different WinGet llama.cpp directory in `PATH` is a
newer Vulkan set and was not copied into, substituted for, or loaded by the
production process.

Changes made to the binary set: **none**.

## CUDA

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Driver | 616.92 |
| Compute capability | 12.0 |
| VRAM visible to llama.cpp | 8,123 MiB |
| Backend | `ggml-cuda.dll` from the production runtime directory |
| Initialization | PASS; one CUDA device enumerated, VMM enabled |

`nvidia-smi` saw the GPU with free VRAM and no stale compute process. A sibling
runtime probe printed CUDA device enumeration and loaded the production
`ggml-cuda.dll`. The production server then loaded the model and generated
normally. There is no evidence of a CUDA initialization failure in the blocked
historical launches; Code Integrity stopped them earlier.

## GGUF

| Field | Value |
|---|---|
| Path | `models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf` (ruta relativa al repo) |
| Size | 5,168,653,536 bytes |
| SHA256 before | `7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429` |
| SHA256 after | `7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429` |
| Unchanged | YES |

## Launch and readiness

The selected profile remained `qwen35-9b-iq4-xs`. The exact relevant production
command is:

```text
llama-server.exe --model models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf --alias Qwen3.5-9B-IQ4_XS --ctx-size 8192 --parallel 1 --host 127.0.0.1 --port 8081 --flash-attn off
```

The direct command loaded the model in about 4.76 seconds, created one slot with
`n_ctx_slot=8192`, and listened on `http://127.0.0.1:8081`. `/health` returned
`{"status":"ok"}` and `/v1/models` advertised the expected alias, context 8192,
and IQ4_XS model metadata. A deterministic small completion request returned
HTTP 200; the later real JARVIS smokes produced visible deterministic text.

`LocalModelRuntimeManager` independently launched the same command from the
binary directory, reported CUDA readiness with JARVIS ownership, and later
stopped its own process normally. Final process and listener checks found no
stale `llama-server` and nothing listening on 8081.

## Fix

- Production source changed: **NO**.
- Runtime executable or DLL changed/replaced: **NO**.
- GGUF changed: **NO**.
- Windows policy changed by this task: **NO**.
- Driver, VC runtime, CUDA runtime, PATH, context, parallelism, or provider
  configuration changed: **NO**.
- Evaluation additions:
  - `training/agentic_lora/evaluation/run_p33r_live_smokes.py`
  - `training/agentic_lora/evaluation/run_p33r_external_research.py`
  - `training/agentic_lora/data/p33r_live_smokes.json`
  - `training/agentic_lora/data/p33r_external_research.json`

The recovery was verified rather than fabricated by replacing a healthy pinned
set. If the Code Integrity policy rejects the unsigned DLL again, the concrete
next diagnostic is the same Operational log and policy ID; a durable signing or
allow-policy decision would require explicit system-administration authority.

## Live smokes

### A — Simple information: PASS

| Metric | Result |
|---|---|
| PRE-LORA profile | `full` |
| Prompt tokens | 5,367 |
| Initial output budget | 1,024 |
| Segments | 1 |
| Finish reasons | `stop` |
| Completion tokens | 408 |
| Tool executions | 0 |
| Latency | 9.383 seconds |
| Visible completion | coherent, non-empty, no truncation |

The ordinary question was answered without an unnecessary tool call. No
continuation was needed.

### B — Files action: PASS

| Metric | Result |
|---|---|
| PRE-LORA profile | `full` |
| Prompt tokens on final call | 6,054 |
| Schemas supplied | 10 relevant schemas |
| Selected tool | `files.search` |
| Tool executions | exactly 1 |
| SecurityGate | `allowed`, safe |
| Observation | success, `verified=true`, fixture observed |
| Segments/model calls | 2: `tool_calls`, `stop` |
| Budgets | 646 initial, 1,510 final |
| Latency | 19.939 seconds |
| Final grounding | exact observed path retained; no invented path |

No continuation was necessary, and the Files side effect was not repeated.

### C — Forced long response: PASS

| Metric | Result |
|---|---|
| PRE-LORA profile | `full` |
| Prompt tokens | 4,652 |
| Initial budget | 917 |
| Segment budgets | 917, 1,024 |
| Segment completion tokens | 917, 566 |
| Finish reasons | `length`, `stop` |
| Continuations | 1 |
| Initial schemas | 12 |
| Continuation schemas | 0 |
| Continuation `tool_choice` | `null` |
| Tool executions | 0 |
| Duplicated start | NO |
| Raw reasoning exposed | NO |
| Explicit final marker present | YES |
| Latency | 43.953 seconds |

The two provider segments were merged into one logical visible assistant answer.

Machine-readable evidence is stored in
`training/agentic_lora/data/p33r_live_smokes.json` and can be regenerated with
`python -m training.agentic_lora.evaluation.run_p33r_live_smokes`.

## Deep Research

### Deterministic: PASS

The P3.3 diagnostic still reports stable S1/S2 identities, stable synthesis
prompt, verified links, zero promoted unknown citation IDs, bounded stopping,
and `length -> stop` completion with tools removed. The focused P3.3 suite also
passes.

### External non-Brave: PASS

The configured non-Brave providers were isolated in a router containing only
You and Exa. Brave was not present and received zero requests.

| Metric | Result |
|---|---|
| Providers | You: 4 requests; Exa: 4 requests |
| Search rounds | 1 |
| Sources | 12 across multiple domains |
| Stop reason | `evidence_sufficient` |
| Synthesis attempts | 3 bounded attempts |
| Text continuations | 1 before citation validation |
| Final status | `completed_with_repairs` |
| Citation validation | PASS; zero invalid IDs |
| Source-only fallback | NO |
| Latency | 66.623 seconds |

The detailed result is in
`training/agentic_lora/data/p33r_external_research.json` and is reproducible via
`python -m training.agentic_lora.evaluation.run_p33r_external_research`.

## Regression and invariants

- Local runtime/profile tests: **40 passed**.
- P3.3 focused suite: **239 passed**.
- Deterministic P3.3 diagnostics: **PASS**.
- Ruff for added Python runners: **PASS**.
- `git diff --check`: **PASS**; repository-native LF/CRLF warnings only.
- PRE-LORA-2 full: **UNCHANGED**, with the prior 3,169-token static prompt and
  production capability-selection behavior intact.
- Brave: **UNCHANGED by P3.3R; files touched: none**.
- Frozen B3: **UNCHANGED; no diff in manifest, canonical/compiled datasets,
  frozen benchmark, or baseline artifacts**.
- B4A/B4A.2: **UNTOUCHED**.
- Training: **NOT STARTED**.
- Cloud model: **NOT USED**.
- Observed model/training cost: **$0**. External search-provider billing is not
  exposed by the runtime and is not guessed.
- Unrelated and concurrent worktree changes were preserved.

## Promotion

**P3.3 RESULT: PASS.**

**READY FOR PHASE 4: YES.**

Recommended next phase: proceed to the planned Phase 4 gate while retaining the
new P3.3R runners as preflight checks. Treat any recurrence of `0xC0E90002` as a
host Code Integrity/signing incident, not as a CUDA, GGUF, context, or launcher
failure unless new evidence moves the first failing layer.
