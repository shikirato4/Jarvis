# PRE-LORA-3 evaluation and production decision

## Result

**PARTIAL.** The provider-neutral registry metadata and bounded retrieval path are implemented. The runtime is not yet a definitive universal selector: a second, genuinely unseen paraphrase set retrieved the exact intended schema set in only 5 of 10 cases. Compact also executed an information-only Windows prompt in one of seven fake-backend cases. Model-driven copy/move and error recovery across all requested Files trajectories remain unmeasured.

## Architecture and first failing layer

`ActionRegistry` definitions are the executable source of truth. `ActionExecutor.tool_schemas()` exposes their callable schemas and `tool_semantics()` exposes separate semantic metadata. `AgentController.prepare_model_request()` passes both to `AgenticPromptPolicy`, which ranks and budgets disclosures before a provider serializes them. Tool calls still go through `SecurityGate`, `ActionExecutor`, structured observations, and the existing controller loop. No second executor or model-specific semantic router was added.

The first failing layer in the original request was retrieval: the previous cutoff (`max(3, max_score-2)`) and English-biased lexical scores removed relevant Files and Windows operations before the model saw them. Missing companion metadata also removed later steps. The model has a separate decision failure: with `web.search` supplied, the compact model searched once for plural sources, while full searched three times; neither matched the specified two-search trajectory.

## Retrieval design

- Definitions declare semantic aliases and companion names for Files, Windows, Terminal and Web operations. The selector uses normalized Unicode tokens, operation/family metadata, alias overlap, direct intent, and bounded companion expansion. Unknown tools remain usable through their generic schema/name/description and related same-family operation suffixes.
- Action versus information remains a model decision. The selector may disclose a tool for a how-to question; it does not execute it. The fake Windows and baseline model trials checked this distinction.
- The full family manifest is generated from the active registry. Detailed schemas are disclosed selectively, with a configurable count limit of 24 and estimated serialized schema budget of 2,400 tokens for compact, 1,800 for full. The estimate is conservative UTF-8 bytes/4; the evaluation also measures actual Qwen chat-template tokenization.
- This is a lexical and declarative multilingual strategy. It is extensible to additional languages by metadata, but the 5/10 unseen result shows that Spanish/English alias coverage and precision are still insufficient for a definitive runtime. A semantic retriever should be evaluated against a larger blind set before promotion to PASS.

## A/B evidence

All numbers below come from the local Qwen3.5-9B-IQ4_XS runtime at temperature 0, seed 42. The 14 model cases use the repository's existing PRE-LORA-2 development cases and simulated observations. Candidate cases select from the real 105-schema registry; the synthetic unknown-tool schemas are added where a case requires them. The two false-execution rows were rerun after fixing unknown-operation retrieval. Latency varies with local runtime startup and is not a reliable profile discriminator.

| Arm | Exact trajectory | First tool | Arguments | Mean cumulative prompt tokens | Mean latency |
|---|---:|---:|---:|---:|---:|
| Current full | 13/14 | 14/14 | 14/14 | 7,426.43 | 11,620.78 ms |
| Current compact | 13/14 | 14/14 | 14/14 | 2,070.79 | 26,879.17 ms |
| Candidate full | 13/14 | 14/14 | 14/14 | 7,668.79 | 7,930.72 ms |
| Candidate compact | 13/14 | 14/14 | 14/14 | 2,307.43 | 8,050.79 ms |

Candidate retrieval on 23 development prompts: full and compact both had 1.00 required-tool recall and 1.00 precision, versus current full 0.3043 recall / 0.2699 precision and current compact 0.3478 / 0.2699. Mean detailed-schema count fell from 3.87/5.17 to 1.35; mean actual schema tokens fell from 796.57/1,067.26 to 260.09. Mean rendered request tokens were 4,000.13 for candidate full and 1,370.13 for candidate compact. Maximum rendered request tokens were 4,173 and 1,543 respectively; the baseline full maximum was 7,997, leaving inadequate room within 8,192 for output. The rendered request count excludes conversation history and output reserve; production context compaction still applies.

The first new paraphrase set initially scored 2/10 and was used to expand declarative aliases, reaching 10/10; it is therefore development evidence only. The subsequent untouched set scored **5/10**, with false positives from other active families and missed Files/Windows operations. It is the controlling generalization result.

The 14-case candidate model matrix had valid arguments in all cases, no unnecessary tool calls, no false-success claims after a failed observation, and both explicit Files multi-step trajectories passed. Compact produced a nonempty final in 14/14; full in 13/14. The only trajectory mismatch for each profile was the plural-sources research case. A four-case removable few-shot probe improved that case for compact without harming the other three, but is too small to justify retaining the examples in production.

**Production profile:** `full` remains the canonical default in `Settings` and `AgenticPromptPolicy`; `off` and `compact` remain configurable. Compact matched full on the 14-case exact trajectory, produced one more nonempty final, and used about 70% fewer cumulative prompt tokens. A separate fake-backend Windows run passed 7/7 with full but only 6/7 with compact: compact called `windows.environment.inspect` for “¿Cómo puedo ver qué versión de Windows tengo?” This action-versus-information regression blocks compact promotion. Existing explicit environment configuration can override the default; `.env` was not edited.

## Integration and limits

The in-memory Windows model matrix passed 7/7 action/information prompts under full. Compact passed 6/7 on the second run; the first run hit runtime failures after two cases and is not used as behavioral evidence. No native Windows action was performed. Isolated Files tests verified search→read, search→metadata, search→copy→verify, and search→move→verify through `SecurityGate`/`ActionExecutor`, including confirmation for copy/move, plus no-result, invalid-argument, and missing-target outcomes. The existing model matrix verified search→read and search→metadata only; it does not prove the model completes copy/move or every failure case.

The focused P3.3R/agent/Files/Windows/provider suite passed 242 tests after these changes. Ruff and `git diff --check` passed. The pre-existing dirty web/research, chat, and output-completion files were preserved and their focused tests were included. GGUF, llama.cpp, CUDA, context 8192, Brave, frozen B3, B4A/B4A.2, LoRA training, and cloud GPU were untouched.

## Next gate

Improve retrieval precision and multilingual coverage using a blind, versioned paraphrase benchmark that includes ambiguous cross-family tools; repair compact's Windows information control before any future promotion; then rerun the 14-case model matrix and model-driven Files copy/move/failure trajectories. Keep semantic routing in the registry/policy, with provider adapters limited to serialization. Only declare PRE-LORA-3 PASS after the blind set and full Files trajectories pass without profile or context regressions.
