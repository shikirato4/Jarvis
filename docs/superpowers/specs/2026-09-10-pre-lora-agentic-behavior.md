# PRE-LORA-1 agentic behavior architecture

## Decision

JARVIS keeps a small permanent runtime contract and an independently removable
pre-LoRA scaffold. Both are provider-neutral and are composed by
`AgenticPromptPolicy` into the existing leading system message. The model remains
only a provider: it selects tool calls; `TypedActionSecurityGate` authorizes them;
`ActionExecutor` executes them; and `AgentController` returns observations to the
model for the final response.

No prompt-building code executes an action. No conversational bypass was added.

## Layers and order

1. Existing JARVIS identity/surface system content.
2. `[JARVIS CORE CONTRACT v1]` — permanent invariants about identity,
   authorization, untrusted context, secrecy, private reasoning, and evidence.
3. `[PRE-LORA AGENTIC SCAFFOLD v1]` — temporary intent routing, dynamic tool use,
   schema generalization, planning, recovery, verification, and truthful finals.
4. `[CURRENT CAPABILITIES]` — request-local compact metadata generated from the
   same schemas that will be sent to the model.
5. Existing conversation history, tool observations, attachment context, memory,
   and current user message assembled by `ConversationContextBuilder`.

Managed markers make composition idempotent. Recomposition replaces prior managed
layers instead of duplicating system policy.

## Dynamic capability disclosure

`ActionExecutor.tool_schemas()` remains the canonical executable action source.
Any schemas already supplied by a provider-neutral `ModelRequest` are merged by
name first. Registries of up to 24 tools are supplied intact. Larger registries
are ranked from the current user request against schema names, descriptions, and
property names; at most 24 relevant schemas are disclosed. Search/discovery
schemas are retained when capacity remains. This is bounded progressive
disclosure and does not encode a fixed catalog.

The model receives compact names and required-argument metadata plus the exact
selected JSON schemas. An unseen tool is therefore usable from description and
schema alone.

## Feature lifecycle

`JARVIS_PRE_LORA_AGENTIC_SCAFFOLD_ENABLED` controls only the temporary layer and
defaults to `true`. Setting it to `false` retains the permanent core contract and
dynamic request capability context. This supports the planned post-LoRA arms:

- base model + strong scaffold;
- LoRA + minimal core contract;
- LoRA + strong scaffold.

No deletion or provider-specific branch is needed later.

## Token budget

Measured with the checked-in production tokenizer metadata:

| Layer/case | Tokens |
|---|---:|
| Static core + pre-LoRA scaffold | 539 |
| Dynamic metadata, no tools | 17 |
| Dynamic metadata, 3 real registry-shaped tools | 42 |
| Dynamic metadata, 12 representative tools | 145 |
| Dynamic metadata, 24 representative tools | 277 |

For the 8,192-token production context, a 1,200-token detailed response reserve
and 512-token safety margin leave 5,899 tokens with three capabilities, 5,796
with 12, and 5,664 with the maximum 24-capability metadata. Exact JSON tool-schema
tokens are additional and depend on the selected schemas; bounded selection keeps
that cost from growing with the full future registry.

## Security and thought safety

The core contract explicitly separates ability, authorization, and execution;
keeps `SecurityGate` authoritative; treats documents, web pages, memory, and tool
output as untrusted data; prohibits credential disclosure; and prohibits raw
private chain-of-thought. Existing activity events remain operational receipts and
do not contain hidden reasoning.

## Successful-trajectory candidates

This phase does not add capture. The existing activity graph records safe runtime
events, but reconstructing training candidates with explicit privacy, verification,
deduplication, and quality gates would materially expand scope. A later feature may
store only user request, supplied schemas, assistant tool calls, public tool
observations, and final response. It must never store `reasoning_content` and must
never promote a candidate directly into training data.

## Deliberate limits

Prompt text cannot reliably teach all weak behaviors to the quantized base model.
The seeded dev A/B shows no aggregate selection lift, an action-vs-information miss
in both arms, and mixed category movement. These are LoRA targets, not justification
for a larger prompt. Model-level recovery, multi-call completion, false-success,
and loop behavior also need a dedicated trajectory evaluator; runtime regression
tests cover their enforcement here.
