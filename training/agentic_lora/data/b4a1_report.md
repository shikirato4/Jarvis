# JARVIS Agentic LoRA B4A.1 — llama.cpp Qwen3.5 LoRA compatibility gate

Status: **PARTIAL — production runtime proved; official converter remains blocked**

Decision: **UPSTREAM_FIX_REQUIRED**. The production b10516 runtime can load and
apply a structurally valid Qwen3.5 GGUF LoRA. Both the exact b10516 converter
source and current official master fail when converting the planned
`linear_attn.out_proj` adapter. No production binary update is presently
needed; a reviewed converter fix is needed before cloud spend.

## Exact builds inspected

- Production binary/source: `b10516-b95502ba9`, full commit
  `b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9`.
- Current official master: `b96806d96061049a5b574269b049bf6241d63d46`.
- `convert_lora_to_gguf.py` and `src/llama-adapter.cpp` are byte-identical
  between these snapshots. Current master has no fix for the reproduced
  Qwen3.5 LoRA output-projection failure.
- The production binary exposes `--lora`, `--lora-scaled`, and server
  `--lora-init-without-apply`. The matching source contains
  `convert_lora_to_gguf.py` and `llama-export-lora`; those two tools are not in
  the installed production binary bundle.

## Production GGUF

The unchanged production file has `general.architecture=qwen35`, 427 tensors,
32 blocks, embedding 4096, FFN 12288, and
`qwen35.full_attention_interval=4`. Linear-attention layers are 0–2, 4–6,
8–10, 12–14, 16–18, 20–22, 24–26, and 28–30. Full-attention layers are 3, 7,
11, 15, 19, 23, 27, and 31. SHA-256 before and after the test remained
`7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429`.

## PEFT to GGUF mapping

| PEFT logical target | GGUF base tensor | Count | Official conversion |
|---|---|---:|---|
| `linear_attn.in_proj_qkv` | `blk.{layer}.attn_qkv.weight` | 24 | PASS |
| `linear_attn.in_proj_z` | `blk.{layer}.attn_gate.weight` | 24 | PASS |
| `linear_attn.in_proj_b` | `blk.{layer}.ssm_beta.weight` | 24 | PASS |
| `linear_attn.in_proj_a` | `blk.{layer}.ssm_alpha.weight` | 24 | PASS |
| `linear_attn.out_proj` | `blk.{layer}.ssm_out.weight` | 24 | **FAIL** |
| `self_attn.q_proj` | `blk.{layer}.attn_q.weight` | 8 | PASS |
| `self_attn.k_proj` | `blk.{layer}.attn_k.weight` | 8 | PASS |
| `self_attn.v_proj` | `blk.{layer}.attn_v.weight` | 8 | PASS |
| `self_attn.o_proj` | `blk.{layer}.attn_output.weight` | 8 | PASS |
| `mlp.gate_proj` | `blk.{layer}.ffn_gate.weight` | 32 | PASS |
| `mlp.up_proj` | `blk.{layer}.ffn_up.weight` | 32 | PASS |
| `mlp.down_proj` | `blk.{layer}.ffn_down.weight` | 32 | PASS |

The full mapping is known, but the deployment mapping gate is PARTIAL because
the official converter cannot serialize one selected target. Silently dropping
`linear_attn.out_proj` is not acceptable.

## Reproduced converter failure and structural proof

A temporary rank-1, zero-effect PEFT-shaped fixture covered one representative
module from every planned family. Both unmodified converter snapshots failed
at `linear_attn.out_proj`: Qwen3.5 reorders the projection's input columns, and
`LoraTorchTensor.reshape()` rejects changing the represented row dimension.
This is the exact failure tracked by llama.cpp issue #21125.

In the isolated current-master checkout only, the reshape/permute/reshape was
replaced with its mathematically equivalent direct index permutation. A random
tensor equivalence assertion passed, all 12 adapter families converted to a
24-tensor BF16 LoRA GGUF, and the unmodified production b10516 CLI loaded the
adapter with the production IQ4_XS base. CUDA generation completed successfully
with exit code 0 at 47.4 tokens/s. The fixture was
`UNTRAINED_PREFLIGHT_ONLY`, all adapter values were zero, and the temporary
converter patch was neither installed nor committed.

All owned `llama-cli` processes were stopped. Recursive and individual file
removal were both rejected by the execution safety layer after exact-path
verification, so the isolated test root remains at
`%TEMP%/jarvis-b4a1-c3d7367484b14e40a357d09821660f8e` (approximately 415 MB).
It contains only the two official source checkouts, structural fixtures, test
GGUF adapters, and logs; it is not referenced by JARVIS or production.

## Known issue impact

- [#21125](https://github.com/ggml-org/llama.cpp/issues/21125): **directly
  applicable and reproduced** for this exact Qwen3.5-9B target set at
  `linear_attn.out_proj`.
- [#20450](https://github.com/ggml-org/llama.cpp/issues/20450): older missing
  `contiguous()` failure; fixed before b10516 and not the present blocker.
- [#27019](https://github.com/ggml-org/llama.cpp/issues/27019): concerns full
  HF-model conversion/merged tensors. The production GGUF already loads, and
  structural LoRA conversion of `in_proj_a`/`in_proj_b` passed, so it is not the
  demonstrated adapter blocker here.
- [#28207](https://github.com/ggml-org/llama.cpp/issues/28207): concerns
  non-uniform `layer_types`. The exact official 9B production artifact has a
  regular interval-4 layout confirmed by tensor inventory and successful
  runtime load; it does not block this base.

## Converter requirements and future command

The converter requires a PEFT directory containing `adapter_config.json` plus
`adapter_model.safetensors` (or `.bin`). With explicit `--base`, only the pinned
HF config/tokenizer metadata is required; full HF base tensors are not required.
After an official or separately reviewed fix exists, the command template is:

```text
python <compatible-llama.cpp>/convert_lora_to_gguf.py \
  --base <Qwen3.5-9B-pinned-config-and-tokenizer-dir> \
  --outfile <jarvis-agentic-lora-bf16.gguf> \
  --outtype bf16 \
  <peft-adapter-dir>
```

Runtime validation remains:

```text
llama-cli -m Qwen3.5-9B-IQ4_XS.gguf --lora jarvis-agentic-lora-bf16.gguf ...
```

The base stays unchanged. `llama-export-lora` is secondary only and is not
needed for the preferred standalone-adapter deployment.

## Final report

**B4A.1:** PARTIAL

**PRODUCTION LLAMA BUILD:** `b10516-b95502ba9`

**CURRENT UPSTREAM COMMIT INSPECTED:**
`b96806d96061049a5b574269b049bf6241d63d46`

**PRODUCTION QWEN35 SUPPORT:** PASS

**PRODUCTION --LORA SUPPORT:** PASS

**CONVERT_LORA_TO_GGUF:** FAIL for the complete planned target set

**QWEN3.5 CONVERTER SUPPORT:** FAIL — `linear_attn.out_proj` is unsupported

**QWEN3.5 KNOWN ISSUE IMPACT:** #21125 directly applies; other reviewed issues
do not block this exact standalone-adapter path

**PRODUCTION GGUF ARCH:** `qwen35`

**HYBRID LAYOUT:** 32 blocks; 24 linear-attention and 8 full-attention layers;
full-attention interval 4

**PEFT → GGUF TENSOR MAPPING:** PARTIAL — 12 mappings resolved, 11 convert with
unmodified official tooling

**UNRESOLVED TARGET MODULES:** `linear_attn.out_proj`

**SYNTHETIC ADAPTER TEST:** PASS with isolated diagnostic converter fix

**GGUF ADAPTER LOAD:** PASS

**TINY INFERENCE:** PASS

**PRODUCTION BASE MODIFIED:** NO

**RECOMMENDED DEPLOYMENT:** pinned base GGUF plus standalone BF16 LoRA GGUF via
`--lora`, after a reviewed/upstream converter fix and real cloud PEFT round-trip

**PRODUCTION LLAMA UPDATE REQUIRED:** NO for runtime; converter fix required

**MINIMUM COMPATIBLE BUILD:** current b10516 runtime; official compatible
converter commit unresolved

**REAL CLOUD PEFT ROUND-TRIP STILL REQUIRED:** YES

**CLOUD STARTED / CLOUD COST / TRAINING STARTED:** NO / $0 / NO

**B3 FROZEN:** PASS

**TESTS:** 19 passed in 6.10 seconds (`training/agentic_lora/tests`)

**RUFF:** PASS

**GIT DIFF CHECK:** PASS

**READY TO AUTHENTICATE CLOUD AND RESUME B4A:** NO — resolve and review the
converter fix first

**BLOCKERS:** official llama.cpp issue #21125 for Qwen3.5
`linear_attn.out_proj`; authenticated cloud access remains separately absent.
Owned temporary files remain because deletion was rejected by the execution
safety policy; no owned process or port remains.

Machine-readable evidence:
`training/agentic_lora/data/b4a1_compatibility_snapshot.json`.
