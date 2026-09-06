# JARVIS Agentic LoRA: Architecture, Feasibility, and Dataset Contract

**Date:** 2026-08-30  
**Status:** B0/B1 design complete; no training authorized  
**Production baseline:** Qwen3.5-9B-IQ4_XS, profile qwen35-9b-iq4-xs, qwen_local, 8,192 context, parallel 1

## 1. Decision summary

The first JARVIS LoRA should be a narrowly targeted behavioral adapter that teaches a model how to use capabilities supplied at runtime. It must not memorize a permanent catalog of JARVIS tools. The runtime registry and SecurityGate remain the authorities for capability availability and action authorization.

The production IQ4_XS GGUF is an inference artifact and is not the correct input to conventional PEFT training. Training should begin from the exact upstream post-trained Qwen/Qwen3.5-9B Transformers checkpoint whose tokenizer, chat template, configuration, and behavior are verified against the production GGUF. The preferred pilot is BF16 LoRA on a temporary Linux GPU with 48 GB VRAM. Four-bit QLoRA on the current 8 GB laptop is technically close to some framework minimums, but Qwen3.5-specific guidance warns against it and the measured free system memory leaves too little operating margin. Local training is therefore classified LOCAL_NOT_WORTH_IT for this project; WSL2 remains useful for tooling and small smoke tests, not the promotion training run.

The canonical dataset will be versioned JSONL with canonical dotted tool names, structured messages, exact tool schemas, exact observations, and evaluation-only metadata. A deterministic compiler will produce the production-near Qwen surface: registry-derived capability instructions, selected OpenAI-compatible JSON schemas, llama.cpp-safe double-underscore tool names, official pinned Qwen3.5 chat rendering, and assistant-only loss labels. No private chain-of-thought or reasoning_content is stored or trained.

The pilot recommendation is 1,200 high-quality records, followed by a 5,000-record v1 only if the pilot identifies coverage gaps. Training/validation data should contain 25% current real-tool cases, 60% fictitious training-tool cases, and 15% non-tool conversational controls. Completely separate fictitious families are reserved for the unseen-tool hard gate.

## 2. Inspection scope and production architecture

This design is based on the current repository rather than a generic tool-calling format.

### 2.1 Request construction

ConversationContextBuilder constructs provider-neutral ModelMessage sequences. It deduplicates and budgets history while preserving the system message, operational tool history, attachments or memory, and the current user request. DesktopChatEngine.finalize_model_request separates tool calls and tool observations from ordinary conversational history before invoking the builder.

The core model-facing types are:

- ModelMessage with role, content, optional name, tool_call_id, and tool_calls.
- ModelToolCall with id, name, and arguments.
- ModelRequest with messages, tools, tool_choice, and generation settings.

DesktopChatEngine finalizes the request and passes it into AgentController. AgentController injects the actual ActionExecutor tool schemas, sets automatic tool choice, and adds a capability-awareness system instruction derived from the current registry. That instruction distinguishes an explicit action request from an informational question and requires a concise, observation-grounded, non-empty final answer after execution.

### 2.2 Runtime tool registry and schemas

The agent-facing source of truth is the ActionRegistry exposed through ActionExecutor, not a model-owned list and not the separate generic ToolRegistry abstraction. Each ActionDefinition provides:

- canonical dotted name;
- description;
- Pydantic payload model;
- handler;
- risk classification;
- timeout and cancellation behavior;
- idempotency and rollback properties;
- adapter metadata where applicable.

ActionExecutor.tool_schemas serializes each supplied capability using the OpenAI-compatible function form:

~~~json
{
  "type": "function",
  "function": {
    "name": "namespace.action",
    "description": "What the action does",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }
}
~~~

The JSON Schema emitted by the payload model is authoritative for names, types, enums, and required fields.

### 2.3 Security and execution

TypedActionSecurityGate validates the normalized action and arguments before execution. Terminal and file policies, risk decisions, denials, and confirmation requirements are enforced deterministically in code. Confirmation grants are one-use and fingerprinted to the exact action plus arguments. The model may learn to ask for or respect confirmation, but the adapter cannot grant authority, bypass a denial, or replace SecurityGate.

AgentController executes structured tool calls through ActionExecutor. It enforces a maximum step count, repeated-call signature limits, deadlines, cancellation, and confirmation pause/resume behavior. A denied or failed action is an observation to ground the next assistant turn, not permission to invent success.

### 2.4 Tool-call wire encoding

QwenLocalProvider sends standard OpenAI-compatible messages to llama.cpp. Assistant tool_calls contain a call id, function name, and JSON-string arguments. Tool messages carry the matching tool_call_id and name.

Canonical JARVIS action names contain dots. Before sending a request, QwenLocalProvider maps dots to double underscores in:

- the supplied tool definitions;
- prior assistant tool calls;
- explicit tool_choice values.

It maps returned double-underscore names back to canonical dotted names before execution. Training source records should keep canonical dotted names. The compiler, not authors or generators, must apply the reversible wire transform.

When more than 16 tools are registered, the provider ranks the registry against the current request and supplies at most 8 schemas, with special handling for existing files, terminal, and calculator capabilities. Dataset records therefore declare the model-visible tool subset, not an imaginary global catalog. The dedicated evaluation needs both a model-surface track with already-selected schemas and an end-to-end track that includes current runtime selection.

### 2.5 Observation representation

After every action, AgentController appends an assistant message containing the structured tool call and then a tool message. The observation content is deterministic compact JSON with sorted keys and compact separators:

~~~json
{
  "action": "namespace.action",
  "status": "success",
  "message": "Human-readable status",
  "data": {}
}
~~~

When applicable it also contains error_type or confirmation_id. Large file results are compacted, including content truncation around 2,000 characters, list limits, and critical metadata preservation. Training examples must use this observation envelope and must link each observation to the exact preceding call id and canonical action.

### 2.6 Continuation and finalization

The observation becomes input to another model continuation. The model may call another supplied tool or give a final answer. Completion handling preserves finish_reason. AgentController treats length termination, missing finish reasons, and stream mismatches as incomplete rather than completed. The normal agent path reserves at least 768 output tokens, and DesktopChatEngine prevents an empty visible final response. These semantics are part of the training and evaluation contract; no adapter result may hide finish_reason=length or convert incomplete work into success.

### 2.7 Thinking adapter

The existing Qwen3.5 reasoning adapter is binary. Off and low request enable_thinking=false, while the thinking-enabled settings preserve the current Qwen behavior. Agentic SFT should render with thinking disabled and never store reasoning_content. Post-training evaluation should exercise the production binary modes, while recording only observable tool calls, observations, final answers, and completion metadata.

### 2.8 Runtime manager and existing evidence

LocalModelRuntimeManager launches llama-server CUDA-first with Vulkan fallback, the selected model alias, context, parallelism, and flash attention settings. The current profile is text-only Qwen3.5-9B-IQ4_XS at 8,192 context and parallel 1. The historical Qwen3.8 profile is irrelevant to LoRA training and is not a dependency.

Existing tests and harnesses provide reusable patterns:

- agent runtime, calculator flow, terminal/files, conversation context, Qwen provider, profile, and runtime-manager tests;
- local_model_ab resumable manifests, JSONL results, finish reasons, observation/continuation checks, and reasoning_content filtering;
- coding_validation safe Python execution and assertion accounting;
- benchmark measured-versus-simulated reporting and redacted metadata.

The neural-router training area is useful as a reproducibility and validation precedent, but its small static lexical tool-id problem is not the Agentic LoRA training surface.

## 3. Exact current training surface

The closest faithful training surface is an observable OpenAI-compatible conversation compiled exactly as follows:

1. Begin with the same registry-derived capability-awareness system policy used by AgentController.
2. Supply only the model-visible OpenAI JSON function schemas selected for that request.
3. Preserve normal system/context and user messages after ConversationContextBuilder-compatible budgeting.
4. Encode assistant tool calls structurally, with arguments JSON-string encoded on the rendered wire.
5. Convert canonical dotted tool names to double-underscore names at render time.
6. Append the exact compact JSON tool observation with matching call id and name.
7. Continue with another tool call or a grounded, non-empty assistant final.
8. Render using the pinned official Qwen3.5 tokenizer and chat template.
9. Apply loss only to observable assistant outputs: tool calls and final answers. Mask system/context, schemas, user text, and tool observations.

The canonical source is not a transcript scraped from JARVIS, and a rendered token stream is not the source of truth. Source JSONL, compiler version, tokenizer revision, rendered artifact hash, and label-mask validation are all versioned.

## 4. Base model identity and trainable checkpoint strategy

### 4.1 Inspected GGUF evidence

The production file was inspected read-only with the repository's llama.cpp server. No model file was changed.

- Path: models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf
- Size: 5,168,653,536 bytes, approximately 4.80 GiB
- GGUF version: 3
- Architecture: qwen35
- Name and basename: Qwen3.5-9B
- Parameter count: approximately 8.95 billion
- Quantization: IQ4_XS, approximately 4.61 bits per weight at file level
- Upstream repository recorded in metadata: Qwen/Qwen3.5-9B
- Tokenizer: GPT-2-style BPE with qwen35 pre-tokenizer, 248,320 tokens
- Chat template: embedded ChatML-style im_start/im_end template with thinking support
- Text architecture: 32 layers, 4,096 hidden size, 12,288 intermediate size, 16 attention heads, 4 KV heads
- Hybrid layout: 24 gated linear-attention layers and 8 full-attention layers

The local llama.cpp build reports adapter command-line options, but this only establishes runtime capability in principle. Exact Qwen3.5 target conversion and numerical behavior remain a B3 preflight.

### 4.2 The GGUF is not the training base

IQ4_XS is a highly quantized inference representation. Conventional Transformers/PEFT LoRA expects trainable modules and autograd-compatible weights, neither of which the production GGUF provides. llama.cpp's native GGUF training path is work in progress, uses much higher-precision state, and is not a realistic route for a 9B model on this laptop.

Required strategy:

1. In B3, resolve the exact official post-trained Qwen/Qwen3.5-9B Transformers revision.
2. Pin the model revision and record hashes for configuration, tokenizer, chat template, and Safetensors.
3. Verify architecture and special tokens against the inspected GGUF metadata.
4. Run a small behavior and tokenization equivalence preflight against the production GGUF.
5. Train LoRA against Qwen3_5ForCausalLM, using only the text backbone.

The generic pretraining “Base” checkpoint must not be substituted. Production is derived from the post-trained Qwen/Qwen3.5-9B repository.

The official [Qwen3.5-9B model repository](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.5 Transformers documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md), and [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) are the primary compatibility references. A revision is not considered selected until B3 records and validates it.

## 5. Training-stack feasibility

The classifications below describe the proposed use, not merely whether a package can be imported.

| Component or path | Classification | B0/B1 finding |
|---|---|---|
| Transformers v5 Qwen3_5ForCausalLM | SUPPORTED | Official model implementation and text-only causal LM are available. |
| PEFT LoRA | LIKELY SUPPORTED | Standard linear targets are present; exact target inventory and save/load require a B3 smoke test. |
| TRL SFTTrainer | LIKELY SUPPORTED | Conversational and pre-tokenized labeled datasets are supported; explicit labels are preferred here. |
| bitsandbytes on current Windows/CUDA generation | SUPPORTED generally | Current documentation includes Windows and modern CUDA/SM support. This does not make Qwen3.5 QLoRA advisable. |
| Qwen3.5 QLoRA | NOT RECOMMENDED for promotion | It is memory-efficient, but model-specific Unsloth guidance warns that 4-bit differences are higher for Qwen3.5. |
| Unsloth Qwen3.5 BF16 LoRA on Linux | SUPPORTED | Model-specific instructions and hybrid-kernel optimizations exist. Validate exact revision and export in B3. |
| Unsloth native Windows pilot | NOT RECOMMENDED | Windows is supported generally, but the 8 GB memory limit and hybrid-kernel complexity leave little stability margin. |
| Axolotl Qwen3.5 | LIKELY SUPPORTED | Recent releases include Qwen3.5 support; it adds configuration and operational complexity unnecessary for the first pilot. |
| llama.cpp adapter inference | SUPPORTED at CLI level | Installed binaries expose LoRA options. Exact Qwen3.5 conversion is still a hard preflight. |
| PEFT-to-GGUF adapter conversion | LIKELY SUPPORTED | llama.cpp provides convert_lora_to_gguf.py and conversion tests; target coverage must be proven for this architecture. |
| llama.cpp native GGUF training | EXPERIMENTAL and NOT RECOMMENDED | The training example is explicitly work in progress and is inappropriate for this hardware/model size. |

Primary references are the [PEFT LoRA API](https://huggingface.co/docs/peft/main/package_reference/lora), [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization), [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes), [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), [TRL PEFT integration](https://huggingface.co/docs/trl/peft_integration), [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune), [Axolotl releases](https://github.com/axolotl-ai-cloud/axolotl/releases), [llama.cpp LoRA converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py), and [llama.cpp training status](https://github.com/ggml-org/llama.cpp/blob/master/examples/training/README.md).

No package is installed and no checkpoint is downloaded in B0/B1.

## 6. Training environment recommendation

### 6.1 Native Windows

Native Windows keeps development close to production and current bitsandbytes releases support Windows. It is nevertheless a poor primary training environment here: the GPU has 8 GB, observed free VRAM was about 7 GB during inspection, total system RAM is about 16 GB with about 7 GB free, and Qwen3.5's hybrid layers benefit from specialized kernels. Offload would compete with the running OS and would be slow and fragile. Classification: NOT RECOMMENDED for the actual pilot, suitable for validators and inference evaluation.

### 6.2 WSL2/Linux on the laptop

WSL2 improves package reproducibility and aligns with the Linux training ecosystem, but it does not add VRAM or RAM. It is useful for a tokenizer/compiler smoke test and perhaps a tiny forward/backward test at sequence length 256. It is not the recommended full training environment. Classification: EXPERIMENTAL for training and useful for setup validation.

### 6.3 Temporary Linux cloud GPU

A temporary Linux instance provides stable CUDA tooling and enough memory for BF16 LoRA without changing the production runtime. A 48 GB GPU is the preferred starting point. A 24 GB GPU may be possible with aggressive checkpointing and small batches but is close to published Qwen3.5 LoRA requirements and leaves little margin.

Only the pinned public base checkpoint, source-controlled training code, and a purpose-built sanitized dataset may be present. Do not upload API keys, .env files, raw conversations, personal files, browser history, email, messages, memory databases, or repository secrets. Download the resulting adapter, manifests, and metrics; destroy the cloud workspace according to provider policy after hash verification.

### 6.4 Recommended hybrid

- Local Windows: authoring, deterministic validation, dataset hashing, base and adapter llama.cpp evaluation, regression harnesses.
- Linux 48 GB cloud: environment lock, BF16 LoRA pilot, checkpoints, training metrics, PEFT adapter export.
- Optional WSL2: reproducibility smoke test only.

This is the best balance of stability, cost, privacy, and rollback.

## 7. Local 8 GB feasibility and memory model

Classification: LOCAL_NOT_WORTH_IT.

This is a project recommendation rather than a claim that one optimizer step is impossible.

Approximate contributors:

| Contributor | Approximate magnitude | Notes |
|---|---:|---|
| BF16 upstream base | 16.7 GiB raw | 8.95B parameters times 2 bytes, before framework allocations. |
| Four-bit base | 4.2 GiB raw | Quantization metadata, scales, buffers, and dequantization workspaces raise practical use to roughly 4.8–5.5 GiB. |
| Rank-16 LoRA weights | about 83 MiB BF16 | Approximately 43.3M trainable parameters with the proposed targets. |
| LoRA gradients | about 83 MiB BF16 | Additional master weights and optimizer state add several hundred MiB depending on optimizer. |
| Activations and hybrid attention workspaces | roughly 1–4+ GiB | Strongly dependent on sequence length, kernels, checkpointing, and batch. |
| CUDA/framework overhead | roughly 0.5–1.5+ GiB | Allocator, kernels, graph, and fragmentation. |

Even a four-bit run sits at or above the useful 8 GB envelope once activations and framework state are included. CPU offload is constrained by the observed free RAM and would make the run slow and unstable. The Unsloth Qwen3.5 guide specifically recommends BF16 LoRA rather than QLoRA because quantization differences are larger for this family. A local four-bit experiment may be useful only as a non-promotional compatibility probe; it is not the recommended source of a production candidate.

## 8. Initial LoRA design

### 8.1 Target modules

Use the text Qwen3_5ForCausalLM backbone with PEFT target_modules="all-linear", but guard it with an explicit inventory assertion. The resolved targets must be only:

- linear_attn.in_proj_qkv
- linear_attn.in_proj_z
- linear_attn.in_proj_b
- linear_attn.in_proj_a
- linear_attn.out_proj
- self_attn.q_proj
- self_attn.k_proj
- self_attn.v_proj
- self_attn.o_proj
- mlp.gate_proj
- mlp.up_proj
- mlp.down_proj

Exclude embeddings, lm_head, normalization layers, any vision tower/projector, and unrelated heads. Fail closed if the exact upstream revision resolves a different set. Covering MLP, full attention, and gated linear attention is appropriate because tool-schema comprehension and structured action generation traverse all three, while omitting embeddings and lm_head reduces catastrophic surface area.

At rank 16 this inventory is approximately 43.28M trainable parameters, under 0.5% of the 8.95B base.

### 8.2 Initial hyperparameters

| Setting | Pilot proposal | Reason |
|---|---:|---|
| Rank r | 16 | Enough capacity for structured behavior across hybrid layers without a large personality rewrite. |
| Alpha | 16 | Unit effective scaling is conservative for a narrow behavioral delta. |
| Dropout | 0.05 | Provides an overfit guard for a small high-quality dataset. Compare 0 only if an optimized kernel requires it. |
| Bias | none | Preserves the base when the adapter is disabled and limits trainable surface. |
| Learning rate | 1e-4 | Conservative midpoint for all-linear LoRA; only sweep 5e-5, 1e-4, and 2e-4 after pilot evidence. |
| Scheduler | cosine decay | Smooth decay for a short run without assuming a long plateau. |
| Warmup | 3% of optimizer steps | Avoids a sharp initial update while retaining most steps for learning. |
| Weight decay | 0.01 | Light regularization; LoRA matrices only. |
| Gradient clipping | 1.0 | Protects short, heterogeneous tool examples from spikes. |
| Epochs | 1 initial, maximum 2 | Evaluate after one epoch; stop on held-out and preservation regression rather than fitting to loss. |
| Sequence length | 1,024 tokens | Fits typical single/multi-tool records and controls activation memory. Expand to 2,048 only after truncation analysis. |
| Micro-batch | 1 per GPU | Predictable memory. |
| Gradient accumulation | 16 | Effective batch of 16 sequences without increasing peak activation memory. |
| Packing | off for pilot | Prevents cross-record mask/boundary mistakes; reconsider only after a token-label validator passes. |
| Gradient checkpointing | yes | Necessary memory reduction and reasonable on a cloud training GPU. |
| Precision | BF16 | Matches model configuration and avoids the Qwen3.5 four-bit training warning. |
| Optimizer | fused AdamW | Straightforward on a 48 GB GPU; paged 8-bit AdamW is a fallback only if measured memory requires it. |

Starting hypotheses are deliberately few. B4 should change one controlled variable at a time, keep seeds and splits fixed, and promote on behavior metrics rather than training loss.

## 9. Behavioral scope and catastrophic-change controls

The desired delta is tool-use policy:

available capabilities + user intent + schemas + observations + security policy → correct observable action behavior.

The dataset should not retrain broad factual knowledge, personality, coding style, or conversational tone. Controls:

- train only assistant tool-call/final spans;
- use short, natural finals rather than verbose tutorials;
- include non-tool controls and paired action-versus-information cases;
- use Spanish and English without teaching a new persona;
- avoid generic reasoning or coding tutorials in the training split;
- limit epochs and rank;
- evaluate the adapter disabled, enabled, and removed;
- retain existing reasoning, debugging, instruction, Spanish, coding, safety, and normal-conversation suites as promotion gates.

## 10. Canonical dataset contract

### 10.1 Storage format

JSONL is appropriate because records are independent, streamable, diffable at small scale, easy to hash, and compatible with deterministic validators. UTF-8 without a byte-order mark is required. One logical record occupies one line. Large rendered arrays and tensors are derived artifacts, not hand-edited source.

### 10.2 Stable source schema

~~~json
{
  "schema_version": "jarvis.agentic_lora.v1",
  "record_id": "train_lab_000001",
  "split": "train",
  "locale": "es-MX",
  "category_tags": ["capability_present", "argument_extraction"],
  "tool_family": "lab",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lab.set_incubator_temperature",
        "description": "Set the target temperature of a simulated incubator.",
        "parameters": {
          "type": "object",
          "properties": {
            "celsius": {"type": "number", "minimum": 15, "maximum": 45}
          },
          "required": ["celsius"],
          "additionalProperties": false
        }
      }
    }
  ],
  "messages": [
    {"role": "system", "content": "Runtime policy version reference"},
    {"role": "user", "content": "Pon la incubadora simulada a 37 grados."},
    {
      "role": "assistant",
      "tool_calls": [
        {
          "id": "call_01",
          "type": "function",
          "function": {
            "name": "lab.set_incubator_temperature",
            "arguments": {"celsius": 37}
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "lab.set_incubator_temperature",
      "tool_call_id": "call_01",
      "content": {
        "action": "lab.set_incubator_temperature",
        "status": "success",
        "message": "Temperature target updated.",
        "data": {"celsius": 37}
      }
    },
    {"role": "assistant", "content": "Listo: la incubadora simulada quedó configurada a 37 °C."}
  ],
  "security_expectation": {
    "decision": "allow",
    "confirmation_required": false
  },
  "expected_behavior": {
    "tool_calls": 1,
    "final_must_be_nonempty": true,
    "success_claim_requires_status": "success"
  },
  "provenance": {
    "kind": "programmatic_seed",
    "source_case": null
  },
  "generation_seed": 104729,
  "template_version": "agentic-v1",
  "privacy_review": "synthetic"
}
~~~

Source arguments and observation content are JSON objects for validation. The compiler serializes them to the exact wire strings. Metadata after messages is excluded from model rendering.

### 10.3 Compiler contract

The deterministic compiler:

1. validates the source record;
2. reconstructs the current registry-derived system instruction from a versioned template;
3. applies the runtime-visible tool subset;
4. converts dotted tool names to double underscores;
5. JSON-serializes function arguments and observation content with stable rules;
6. renders with the pinned Qwen3.5 chat template and thinking disabled;
7. tokenizes without silently truncating;
8. creates labels with -100 outside assistant tool-call and final spans;
9. checks that every intended assistant token is labeled and every prompt/observation token is masked;
10. writes a content-addressed derived artifact and manifest.

Do not rely on an unverified template assistant-mask implementation. Explicit pre-tokenized input_ids and labels are the training artifact.

### 10.4 Observable behavior only

Allowed content:

- system/context;
- supplied tool schemas;
- user request;
- assistant structured tool call;
- tool result/observation;
- subsequent assistant tool call where needed;
- final assistant response;
- evaluation/training metadata excluded from rendering.

Forbidden content:

- private chain-of-thought;
- reasoning_content;
- hidden scratchpads;
- think tags;
- raw internal planner traces.

Short reasoning is allowed only when it is ordinary user-visible explanation necessary to the final answer.

## 11. Dataset taxonomy and coverage

Every record has one or more tags. Categories overlap; quotas are monitored by primary scenario and by all tags.

| Tag | Required behavior |
|---|---|
| action_vs_information | Execute “check my version”; explain how for “how can I check”. |
| capability_present | Use a relevant supplied capability. |
| capability_absent | State the limitation honestly and do not fabricate a call. |
| correct_tool_selection | Choose the one relevant schema among alternatives. |
| argument_extraction | Produce valid required values, types, enums, and units. |
| multi_tool | Use observation A to construct call B, then ground the final. |
| observation_grounding | Make no success claim before a confirming observation. |
| tool_failure | Explain, safely recover, or stop according to the observed error. |
| security_confirmation | Respect confirmation, denial, and exact approved arguments. |
| no_false_success | Never say done after failure, denial, timeout, or incomplete execution. |
| no_false_capability_denial | Do not refuse when the supplied schema can satisfy the request. |
| unseen_fictitious_tools | Infer behavior only from new safe schema/name/description. |
| irrelevant_distractors | Ignore irrelevant schemas, including cases where none applies. |
| parameter_ambiguity | Ask only when a required safety-relevant value cannot be inferred. |
| safe_partial_completion | Report exactly which steps succeeded and which did not. |
| continuation_after_observation | Produce a useful non-empty final after execution. |

Suggested pilot coverage gives at least 40 records to each critical tag, with higher counts for capability present/absent, action-versus-information, argument extraction, observation grounding, and unseen schemas. Multi-label accounting prevents a simple easy-case majority from hiding a sparse failure mode.

## 12. Real, fictitious, and non-tool strategy

### 12.1 Training and validation ratio

- 25% current real JARVIS tools.
- 60% fictitious training tools.
- 15% non-tool conversational controls.

Real tools anchor the adapter to current serialization and observations. Fictitious tools dominate so the adapter must read names, descriptions, and schemas rather than memorize terminal, files, and calculator. Non-tool controls protect normal conversation and the action-versus-information boundary.

The ratio applies to train and validation, not to the held-out-generalization split. No static tool list is placed in a permanent system contract.

### 12.2 Fictitious family partition

Training and validation families:

- lab.*
- warehouse.*
- garden.*
- camera.*

Held-out-only families:

- vehicle_sim.*
- spaceship_sim.*
- robot_lab.*
- network_sim.*

Held-out namespaces, action names, descriptions, argument structures, example values, and paraphrase templates must never appear in:

- train or validation records;
- generator demonstrations;
- rejected/negative candidates used for training;
- schema mutation pools;
- documentation injected into generation prompts.

All tasks remain neutral and safe: simulated settings, inventory lookup, sensor readings, scheduled watering, camera framing, robot-lab simulations, and network simulators. No real vehicle, spacecraft, robot, or network control is implied.

### 12.3 Distractors and schema diversity

Examples vary:

- terse and verbose descriptions;
- nested objects and arrays;
- integers versus numbers;
- booleans and enums;
- optional defaults and required arguments;
- similar names with distinct descriptions;
- irrelevant namespaces;
- no applicable tool;
- localized user language with English schema names.

Schema mutations must preserve semantic validity and avoid merely renaming an identical template.

## 13. Negative and contrastive examples

Bad behaviors are evaluation targets, not SFT completion targets. The pilot must never train the model to emit a bad answer or a label such as “BAD RESPONSE.”

Use three representations:

1. Paired contrastive SFT cases: nearby prompts each have their own correct observable answer, such as action request versus informational question.
2. Evaluation-only negative_candidates sidecar: structured failure types and bad candidate outputs for deterministic scoring. The training compiler rejects this field.
3. Optional chosen/rejected preference JSONL reserved for a later explicitly approved preference-training phase. It is not consumed by the B4 SFT pilot.

Negative failure types include unnecessary terminal use, nonexistent calls, wrong types, premature success, repeated failing calls, continuing after denial, false capability invention or denial, ignored observations, loops, and empty finals.

## 14. Privacy and security contract

Training source must be synthetic or purpose-designed. A real regression may be represented only after transformation into a generic case with no identifying content.

Prohibited inputs:

- raw personal conversations;
- JARVIS memory or memory database;
- .env files, keys, passwords, tokens, cookies, or credentials;
- personal paths, documents, browser history, email, or messages;
- raw logs containing private prompts or tool outputs;
- private repositories not explicitly sanitized for this purpose.

Validators scan for secret patterns, high-entropy credentials, emails, phones, private path conventions, IPs where irrelevant, and disallowed provenance. Human review remains mandatory for any sanitized regression source.

SecurityGate behavior is simulated or referenced through a versioned deterministic policy oracle. The model receives examples of correct sequencing, but runtime code remains authoritative in production.

## 15. Split design and dataset size

### 15.1 Pilot: 1,200 records

- train: 800
- validation: 120
- held-out generalization: 200
- hard regression: 80

### 15.2 V1: 5,000 records

- train: 3,600
- validation: 400
- held-out generalization: 700
- hard regression: 300

The pilot is large enough to cover behavior combinations while remaining reviewable. V1 expands only where baseline and pilot error analysis demonstrates a gap. Hundreds of thousands of templated records would amplify leakage and style drift without evidence of benefit.

### 15.3 Leakage controls

- family allowlists are split-specific and fail closed;
- split assignment occurs before linguistic realization;
- task graphs and semantic intents, not just wording, are split;
- generator seeds and prompt packs are disjoint;
- exact normalized hashes catch duplicates;
- token n-gram and MinHash similarity catch near-duplicates;
- human-authored seeds that share an intent graph cannot cross splits;
- held-out families are searched across all source files and generation artifacts.

The hard-regression split is frozen and versioned. It may gain new cases only in a new dataset version; prior results remain reproducible.

## 16. Deterministic data quality gates

Every accepted record must pass:

1. JSON Schema validity and schema_version support.
2. Unique record id and valid split.
3. Supplied-tool consistency: every call names an available tool; no fabricated tool.
4. Function argument JSON Schema validation, including required fields, enums, bounds, and additionalProperties.
5. Call-id uniqueness and exact observation linkage.
6. Canonical dotted-name to wire-name round-trip.
7. Observation action/status/message/data envelope validation.
8. Final grounding: success claims require a success observation; failures and partial state are reported exactly.
9. Security expectation validation against the versioned policy oracle, including confirmation sequence and denial termination.
10. Step and repeated-call limits.
11. Non-empty final response after completed execution.
12. Capability-present and capability-absent consistency.
13. No reasoning_content, private CoT keys, scratchpads, or think tags.
14. Secret, PII, private-path, and prohibited-provenance scanning.
15. Family leakage, exact duplicate, and near-duplicate checks.
16. Rendered token-budget check with no silent truncation.
17. Token-label mask check.
18. Stable serialization and source/derived hash checks.
19. Tool-selector reachability for end-to-end cases.

Failures quarantine a record with machine-readable reasons. Validators do not silently repair semantic errors.

## 17. B2 data-generation strategy

B2 should implement, but not yet execute at scale:

1. Hand-author a small seed set for each critical behavior and both languages.
2. Define safe fictitious schema grammars and task graphs.
3. Generate schema/task combinations programmatically.
4. Use an LLM only to diversify natural phrasing and descriptions, never as the acceptance authority.
5. Generate paired mutations: action/information, capability present/absent, success/failure, unambiguous/ambiguous, and relevant/distractor.
6. Transform selected real regression patterns into generic synthetic cases under human review.
7. Run deterministic validators and leakage checks.
8. Manually review every pilot record or a documented two-pass review equivalent.
9. Freeze source, compiler, manifests, and hashes.

An LLM-generated record is rejected unless its schemas, calls, observations, final grounding, privacy, split, and rendered labels all validate.

## 18. Frozen baseline evaluation

Before any training, B3 runs the exact production base through a dedicated frozen LoRA benchmark. Reuse local_model_ab manifest/resume and completion accounting, but keep the new case source and scoring independent of training data.

A recommended initial benchmark has 600 cases:

- 240 held-out unseen-family cases;
- 180 core agentic/action-selection cases;
- 80 failure, partial-completion, and security cases;
- 50 output-completion/cap cases;
- 50 normal conversation and preservation controls.

Report, never collapse, at least:

- tool selection accuracy;
- argument exactness and schema-valid argument rate;
- action-versus-information accuracy;
- capability-present success;
- capability-absent honesty;
- multi-tool end-to-end success;
- observation-grounding rate;
- failure-recovery correctness;
- confirmation/security correctness;
- unseen-tool generalization;
- final-response non-empty and completion rates;
- tool-loop rate;
- false-success rate;
- false-capability-denial rate;
- fabricated-tool rate;
- finish_reason distribution, including length.

Run two tracks:

1. Model-surface: a fixed supplied schema subset, normally no more than 8 tools, isolates model behavior.
2. End-to-end: the current QwenLocalProvider tool selector and current production request construction are active.

This separation prevents a selector miss from being incorrectly attributed to LoRA schema comprehension while still measuring production behavior.

## 19. Unseen-tool hard gate

Neither training nor validation may contain vehicle_sim.*, spaceship_sim.*, robot_lab.*, or network_sim.*. The frozen hard gate supplies these new schemas only at evaluation time.

Required thresholds for a promotion candidate:

- at least 85% exact relevant-tool selection;
- at least 95% schema-valid arguments;
- at least 80% end-to-end task success;
- at least 95% observation-grounded finals;
- zero security violations;
- zero fabricated calls;
- zero false success on denied or failed actions.

The LoRA must improve held-out end-to-end success by at least 5 percentage points over base. If base is already at or above 90%, non-inferiority within 2 percentage points is acceptable, but no critical metric may regress. These are initial gates to be frozen before viewing LoRA results.

## 20. Base-versus-LoRA A/B and preservation gates

Run:

- Qwen3.5 base GGUF;
- the same base GGUF plus the candidate adapter.

Hold constant prompts, tool schemas, registry state, SecurityGate behavior, context construction, decoding settings, seed where supported, context limit, output cap, reasoning mode, and llama.cpp build. Run multiple seeds only under a predefined manifest and report paired outcomes.

Promotion requires agentic improvement plus no unacceptable regression in:

- reasoning and debugging parity suites;
- general instruction following;
- Spanish;
- normal conversation;
- safety/security behavior;
- executable completed-code correctness, retaining the existing 21/21 assertion expectation.

For scored preservation sets, an initial non-inferiority band is no more than 2 percentage points below base where sample size supports it. Any security violation, fabricated execution claim, or systematic personality drift is a hard failure regardless of aggregate score.

## 21. Output-completion regression

The benchmark must preserve and report finish_reason. It will exercise:

- an explicit 260-token cap to reproduce sensitivity;
- the normal agent minimum of 768 output tokens;
- an adequate-cap control.

Metrics:

- finish_reason=length rate;
- incomplete structured call rate;
- incomplete final-answer rate;
- empty final after successful execution;
- final response token distribution;
- successful continuation after each observation.

The adapter must not increase the length-finish rate by more than 1 percentage point over base and must introduce no new empty final after a successful action. A length finish is never relabeled as completed.

## 22. Deployment and rollback

### 22.1 Option A: base GGUF plus runtime adapter

Convert the winning PEFT adapter to llama.cpp GGUF format using the matching pinned base configuration. Validate every target layer on the exact llama.cpp build, then run the frozen A/B. Keep the adapter optional and profile-scoped. This provides the cleanest disable/rollback path and avoids a second full model artifact.

### 22.2 Option B: merge and requantize

Merge the adapter into the upstream Transformers checkpoint, export an F16/BF16 model, convert to GGUF, and quantize to an approved deployment quant. This can simplify runtime loading but introduces quantization variance, much higher temporary disk use, and a less transparent rollback. It is a later fallback, not the first pilot.

### 22.3 Option C: retain both base and adapter

Always retain the original production GGUF through evaluation and initial deployment. Store the adapter, manifest, and A/B result alongside it. This is required whether Option A or B is ultimately used.

Recommendation: Option A plus Option C. B7 may add an optional adapter-aware runtime profile only after exact conversion/inference compatibility passes and the candidate wins. The production default remains unchanged until a separately reviewed promotion.

## 23. Storage plan

Approximate additional storage:

| Artifact | Estimate |
|---|---:|
| Official upstream Transformers checkpoint and tokenizer | about 19.3 GB |
| Source/compiled pilot and V1 datasets plus manifests | under 0.1 GB |
| Rank-16 PEFT/GGUF adapter | about 0.09–0.15 GB |
| Optimizer/checkpoint state retained for 2–3 checkpoints | about 1.5–4.5 GB |
| Optional separate four-bit training representation | about 5–7 GB |
| Merged Transformers artifact if later produced | about 19–22 GB |
| Temporary F16/BF16 GGUF | about 18–20 GB |
| Requantized IQ4-class GGUF | about 5 GB |

Adapter-first local retention requires roughly 22–24 GB beyond the production GGUF if the upstream checkpoint is also retained locally. If the cloud cache holds the upstream base and only sanitized data, adapter, and manifests return locally, the immediate local addition is under 1 GB.

A later merge/requantize workflow should budget approximately 65–75 GB of peak additional space for upstream, merged, F16/BF16, quantized, checkpoint, and temporary artifacts. Do not create these artifacts until B7 needs them. Do not delete or modify the current production GGUF.

## 24. Reproducibility manifest

Every run records:

- upstream model identifier and immutable revision;
- configuration, tokenizer, chat-template, and Safetensors hashes;
- production GGUF hash and llama.cpp build;
- dataset semantic version, source hash, compiled hash, and compiler version;
- split seed, generation seeds, and family allowlists;
- training seed;
- exact target-module inventory and trainable parameter count;
- rank, alpha, dropout, bias, learning rate, scheduler, warmup, decay, clipping, epochs/steps, batch, accumulation, sequence length, checkpointing, precision, optimizer, and packing;
- Transformers, PEFT, TRL/Unsloth or Axolotl, PyTorch, CUDA, driver, and kernel versions;
- GPU identity and memory;
- start/end time and duration;
- checkpoint metrics and chosen checkpoint;
- baseline and A/B evaluation manifest hashes;
- export/conversion commands and output hashes.

The manifest contains no credentials, private prompts, raw memory, or cloud access metadata beyond non-secret environment facts.

## 25. Repository layout

The repository already uses docs/superpowers/specs for dated design documents. Keep the design here and place future implementation tooling outside production runtime code:

~~~text
docs/superpowers/specs/
  2026-08-30-jarvis-agentic-lora-design.md

training/
  agentic_lora/
    README.md
    configs/
    schemas/
    data/
      source/
      manifests/
      derived/
    scripts/
    validators/
    eval/
    tests/
~~~

Generated datasets, rendered tensors, checkpoints, upstream weights, and cloud caches need explicit ignore rules before B2/B3. Small schemas, configs, manifests, validators, and sanitized seed records may be versioned. Production provider, runtime manager, SecurityGate, registry, and tool behavior remain untouched during B0/B1.

## 26. Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| Upstream/checkpoint mismatch with GGUF | Pin revision; compare metadata, tokenizer, template, special tokens, and behavior before training. |
| Hybrid linear-attention targets unsupported by PEFT/export | Assert target inventory; run one-step save/load and PEFT-to-GGUF tests in B3. |
| QLoRA degradation | Prefer BF16 LoRA on 48 GB Linux; treat four-bit training as diagnostic only. |
| Tool-name memorization | Fictitious tools dominate; family-disjoint hard gate; schema/name mutations. |
| Templating leakage | Split task graphs before realization; semantic near-duplicate detection. |
| Personality or reasoning drift | Assistant-only labels, rank/epoch limits, non-tool controls, preservation A/B gates. |
| False success/security regression | Deterministic observations and policy oracle; zero-tolerance hard gates; runtime enforcement remains. |
| Tool selector confounds model score | Report model-surface and end-to-end tracks separately. |
| Output truncation worsens | Freeze cap matrix and finish_reason metrics before training. |
| Cloud privacy exposure | Upload only public base plus sanitized data/code; hashes and teardown procedure. |
| Windows instability | Train on Linux; use Windows for data work and exact production evaluation. |
| Disk duplication | Adapter-first deployment; defer merge/F16/requantization artifacts. |

## 27. Phase plan

### B2 — Dataset Builder + Validators + Pilot Dataset

- Add the source JSON Schema, compiler, wire-name transform, observation builder, label-mask validator, privacy scanner, leakage detector, and deterministic tests.
- Hand-author seeds and define safe train-only and held-out-only family grammars.
- Produce and review the 1,200-record pilot.
- Freeze dataset and compiler manifests. Do not train.

### B3 — Base Benchmark Freeze + Training Environment Setup

- Resolve and pin the exact upstream Qwen3.5-9B revision.
- Freeze the 600-case dedicated base benchmark and run the production base.
- Create a locked Linux training environment.
- Run tokenizer/template equivalence, target inventory, one-step BF16 forward/backward, save/load, adapter conversion, and llama.cpp inference preflights.
- Stop if Qwen3.5 hybrid targets do not round-trip.

### B4 — Pilot LoRA/QLoRA Training

- Run the single predefined BF16 LoRA pilot on a 48 GB Linux GPU.
- Evaluate after one epoch; run a second only under the predeclared early-stop rule.
- QLoRA is not the primary candidate and requires a separately labeled experimental run if pursued.
- Export only adapter, manifests, metrics, and approved checkpoints.

### B5 — Base vs LoRA Evaluation

- Run paired base versus base-plus-adapter tests under identical production settings.
- Apply unseen-tool, security, false-success, loop, output-completion, coding, reasoning, Spanish, safety, and normal-conversation gates.
- Produce disaggregated results and a promotion recommendation.

### B6 — Iterate Dataset/Training if needed

- Use error taxonomy, not training loss alone, to select targeted data additions.
- Version new records and preserve the original frozen benchmark.
- Change one training variable at a time and rerun all gates.
- Expand toward 5,000 records only when evidence justifies it.

### B7 — Production Adapter Integration if winner

- Add an optional, rollback-safe adapter profile to the existing llama.cpp/JARVIS runtime only after a separately reviewed implementation plan.
- Preserve the base GGUF and unchanged base profile.
- Canary the adapter, retain A/B selection, and document instant disable.
- Consider merge/requantization only if adapter deployment is proven incompatible or materially inefficient.

## 28. B0/B1 exit criteria

- Current architecture and exact production model surface inspected: met.
- GGUF identity and upstream strategy established: met.
- Training feasibility and environment classified: met.
- LoRA targets and starting hypotheses defined: met.
- Dataset, privacy, validation, split, and unseen-tool contracts defined: met.
- Baseline, A/B, length, deployment, storage, risk, and phase plans defined: met.
- Production code/model changed: no.
- Training stack installed: no.
- Large checkpoint downloaded: no.
- Training or dataset generation started: no.

B0/B1 is ready to close. B2 requires a new explicit instruction.
