# JARVIS Agentic LoRA B4A — BF16 + PEFT preflight

Status: **PARTIAL — stopped at hard gates before cloud spend**

No cloud-provider CLI, provider-token environment variable, or user-approved
SSH GPU target is configured. A direct RunPod console check reached account
creation rather than an authenticated account, so no resource could be safely
provisioned. The current upstream llama.cpp Qwen3.5 PEFT-to-GGUF path also has
a reported open conversion failure in `convert_lora_to_gguf.py` while
reordering Qwen3.5 value heads. The local b10516 production bundle has
`llama-server.exe`, but not the Python adapter converter. In accordance with
the B4A deployment gate, no GPU was provisioned and no full checkpoint was
downloaded.

## Final report

**B4A:** PARTIAL

**CLOUD PROVIDER/ENVIRONMENT:** NOT PROVISIONED — no configured provider CLI
or approved SSH GPU target

**GPU / VRAM / CUDA:** NOT AVAILABLE — cloud execution not started

**PYTORCH:** planned pin `2.13.0+cu130`; same version used for offline meta audit

**TRANSFORMERS:** planned pin `5.16.1`; same version used for offline meta audit

**PEFT:** planned pin `0.20.0`; not installed or executed locally

**OTHER DEPENDENCIES:** planned pins `accelerate==1.14.0`, `datasets==5.0.1`,
`trl==1.12.0`, `safetensors==0.8.0`, `tokenizers==0.23.1`

**UPSTREAM BASE:** `Qwen/Qwen3.5-9B`

**REQUESTED REVISION:** `c202236235762e1c871ad0ccb60c8ee5ba337b9a`

**RESOLVED REVISION:** NOT VERIFIED against the full upstream checkpoint;
local B3 metadata is pinned to the requested revision

**REVISION MATCH:** FAIL — required cloud verification was not executed

**TOKENIZER:** local pinned metadata declares `Qwen2Tokenizer`, vocabulary
248320, `eos_token=<|im_end|>`, `pad_token=<|endoftext|>`, and the Qwen3.5
tool-call chat template; full-checkpoint tokenizer identity was not reverified

**TEXT BACKBONE CLASS:** offline zero-weight meta audit resolved
`transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM`

**TEXT BACKBONE PATH:** `model` on the text-only causal LM instantiated from
`Qwen3_5ForConditionalGeneration.text_config`

**TEXT BACKBONE PARITY:** FAIL — meta architecture is consistent, but actual
pinned BF16 checkpoint parity was not executed

**MODEL TOTAL PARAMS:** 8,953,803,264 estimated from the zero-weight text model

**TARGET MODULES:** `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`,
`linear_attn.in_proj_b`, `linear_attn.in_proj_a`, `linear_attn.out_proj`,
`self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`,
`self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`

**TARGET MODULE COUNTS:** 24 each for the five `linear_attn.*` targets; 8 each
for the four `self_attn.*` targets; 32 each for the three `mlp.*` targets; 248
resolved `Linear` modules total

**MISSING TARGETS:** NONE in the zero-weight meta architecture; actual
checkpoint audit not executed

**EXCLUDED MODULES VERIFIED:** PASS for the offline resolver

**VISION/MULTIMODAL LORA PARAMS:** ZERO in the offline target resolution;
actual PEFT injection not executed

**PEFT INJECTION:** FAIL — not executed

**LORA RANK / ALPHA / DROPOUT:** 16 / 16 / 0.05

**TRAINABLE PARAMS:** 43,278,336 estimated; actual PEFT audit not executed

**TRAINABLE PERCENT:** 0.483351% estimated against the text model

**BASE FROZEN:** FAIL — actual PEFT parameter audit not executed

**BF16:** FAIL — config requests BF16, but full BF16 load was not executed

**BASE GPU MEMORY / PEFT GPU MEMORY / FORWARD PEAK MEMORY / HEADROOM:** NOT
AVAILABLE

**TRAIN RECORDS:** 800

**VALIDATION RECORDS:** 120

**TRAIN HASH MATCH:** PASS —
`c48d05db88b85b2acb15bb5f8b178b7ad38434ca2498c7478a50cedf12af34bd`

**VALIDATION HASH MATCH:** PASS —
`f4994819be616767bc91608162e858684ed5fb515c611b102acc0e1af31039d4`

**ASSISTANT-ONLY MASKING:** PASS for compiled artifact validation

**SAMPLE LABELED TOKENS:** 50, 29, 55, 50, 78

**SAMPLE IGNORED TOKENS:** 365, 326, 368, 364, 463

**FORWARD PASS / GENERATION SMOKE / UNTRAINED ADAPTER SAVE / ADAPTER RELOAD /
ADAPTER ROUND-TRIP:** FAIL — not executed

**LOSS / NAN / INF:** NOT PRODUCED / NOT APPLICABLE / NOT APPLICABLE

**UNTRAINED ADAPTER PATH / SIZE:** NONE / 0 bytes

**LLAMA.CPP LORA PATH:** intended path is PEFT adapter -> official
`convert_lora_to_gguf.py --base <pinned-base> --outtype bf16` ->
`llama-server -m <production-base.gguf> --lora <adapter.gguf>`. B4A.1 proved
that b10516 runtime loading works, but official Qwen3.5 conversion remains
blocked specifically at `linear_attn.out_proj` by upstream issue #21125.

**LLAMA.CPP QWEN3.5 SUPPORT:** FAIL for the required adapter conversion path

**PRODUCTION GGUF MERGED:** NO

**PILOT CONFIG VALID:** PASS for identity, target, and data pins; the dependency
set remains unproved in a real cloud environment

**LEARNING RATE / SCHEDULER / WARMUP:** `1e-4` / cosine / 3%

**EPOCH/MAX STEP PLAN:** one epoch; at most two only after validation
justification; microbatch 1, accumulation 16, sequence 1024, no packing,
gradient checkpointing enabled, eval/save every 25 optimizer steps

**ESTIMATED B4B TRAINING TIME:** UNAVAILABLE until an approved GPU is selected
and real step throughput is measured

**RUN MANIFEST:**
`training/agentic_lora/data/b4a_offline_preflight.json` (offline preflight,
SHA-256 `15157d3fa90b3a31f4a5dc2969d29b8e7be21d0be5227d4fbb41b22ea62ff41d`)

**CLOUD COST SO FAR:** $0

**OPTIMIZER STEPS:** 0

**TRAINING STARTED / TRAINED ADAPTER CREATED:** NO / NO

**B3 BASELINE CHANGED:** NO — benchmark and result hashes remain
`085989ef9480870f5c26e0fc72c0e34089d2bf9876109f884d7b8dfc87b9ef20` and
`8468dc0d5c0e3ab9cc8b737a6e0dd9cce8cc523fb0c00911f46ecb742f9795ec`

**PRODUCTION CODE MODIFIED / PRODUCTION MODEL MODIFIED:** NO / NO by B4A

**RAW COT / RAW REASONING_CONTENT:** 0 / 0

**TESTS:** 13 passed in `training/agentic_lora/tests`

**RUFF:** PASS

**GIT DIFF CHECK:** PASS (existing worktree changes remain untouched)

**READY FOR B4B TRAINING:** NO

**BLOCKERS:** (1) no approved cloud target; (2) the required llama.cpp Qwen3.5
adapter conversion path is not currently demonstrated and has a relevant open
upstream failure. Resolve the deployment blocker before paying for GPU time,
then rerun B4A against the exact pinned checkpoint.

## Offline evidence notes

The text model has 32 layers, hidden size 4096, intermediate size 12288, and
24 linear-attention plus 8 full-attention layers. The estimated rank-16
adapter is about 82.55 MiB in BF16 before serialization overhead. Optimizer and
scheduler steps are exactly zero; no trained or untrained adapter artifact was
created. No baseline, GGUF, production runtime, `.env`, cloud resource, or
training artifact was modified.

References: official llama.cpp
[`convert_lora_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py),
official [LoRA conversion/inference test](https://github.com/ggml-org/llama.cpp/blob/master/tests/test-lora-conversion-inference.sh),
and open [Qwen3.5 conversion issue #21125](https://github.com/ggml-org/llama.cpp/issues/21125).
