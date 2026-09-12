# PRE-LORA-2 full capability-awareness architecture

## Decision

PRE-LORA-2 is a temporary, provider-neutral `full` prompt profile for the local
Qwen runtime. It teaches that JARVIS is the complete runtime—not merely its
language model—while keeping execution truth tied to the current registries,
SecurityGate decisions, ActionExecutor observations, and receipts.

The profile is intentionally larger than PRE-LORA-1. With the checked-in
production Qwen tokenizer it costs 3,169 static tokens, below the 3,200-token hard
ceiling. It remains removable after Agentic LoRA training.

## Prompt layers and authority

The model receives one idempotently managed system prompt in this order:

1. permanent core contract;
2. full temporary identity and behavior charter;
3. status-neutral 200-capability architecture universe;
4. compact current family manifest generated from real request-time schemas;
5. concise schema index for the selected tools;
6. exact selected JSON tool schemas through the normal model request;
7. existing memory, project/source context, history, observations, and user input.

The architecture universe is never evidence that a capability is executable.
Only the current manifest and exact schemas describe what can be used now. The
charter explicitly separates capability, permission, execution, and verification.

## Profiles and compatibility

`JARVIS_PRE_LORA_AGENTIC_PROFILE` accepts `off`, `compact`, or `full` and defaults
to `full`. The existing `JARVIS_PRE_LORA_AGENTIC_SCAFFOLD_ENABLED=false` remains a
backwards-compatible master switch and resolves to `off`.

- `off`: permanent core only;
- `compact`: PRE-LORA-1 core and compact scaffold;
- `full`: PRE-LORA-2 charter and complete conceptual universe.

Under extreme context pressure the context builder may recompose `full` into the
compact emergency representation. It preserves core/security invariants, current
manifest, selected schemas, and user content. Normal requests use `full`.

## Current capability discovery

`ActionExecutor.tool_schemas()` is the canonical executable action source.
Provider-neutral schemas already attached to a request are merged by canonical
name. The family manifest is derived from that complete merged set, so families
remain visible even when their detailed schemas are not initially selected.

For `full`, `AgenticPromptPolicy` ranks schema name, description, properties, and
the current user request, disclosing at most 12 exact schemas. Discovery/search
tools are retained when relevant. The same policy is the sole semantic selection
owner. `QwenLocalProvider` only performs wire-safe naming and serialization; it no
longer independently drops tools selected upstream.

## Context budget

Production-tokenizer measurements for a real registry with 96 executable schemas
and a representative Files request:

| Component | Tokens |
|---|---:|
| Permanent core (`off`) | 132 |
| PRE-LORA-1 (`compact`) static prompt | 539 |
| PRE-LORA-2 (`full`) static prompt | 3,169 |
| Current 96-schema family manifest | 271 |
| Selected 12-schema compact index | 136 |
| Selected 12 exact JSON schemas | 2,348 |
| Complete rendered representative request | 6,177 |
| Remaining in 8,192 before generation | 2,015 |
| Remaining after a 1,024-token output reserve | 991 |

The existing rolling OutputCompletionPolicy is unchanged. Provider code does not
reduce output budgets to compensate for this prompt.

## Safety and truth

The charter instructs the model to search before guessing, try before refusing,
verify before claiming, and diagnose/replan after failure. External files, web
content, repositories, memory, and tool observations remain data rather than user
authority. Raw chain-of-thought is prohibited. A tool call still passes through
TypedActionSecurityGate and ActionExecutor; prompt construction never executes a
tool.

Grounding of Windows paths now compares normalized values from structured file
observations rather than JSON-escaped text. A verified observed path is retained;
an invented path is still removed.

## Evaluation boundary

The PRE-LORA-2 development set is separate from frozen B3. It has 14 deterministic
multi-turn cases spanning Files, Terminal, research, action-versus-information,
present/absent capabilities, unseen schemas, multi-tool continuation, failure
recovery, and false execution. A real local Qwen + AgentController Files smoke is
also persisted. No optimizer, training process, cloud provider, or frozen B3 file
is involved.
