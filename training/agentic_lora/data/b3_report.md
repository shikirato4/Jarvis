# B3 report

Status: PARTIAL

The tokenizer audit, production-template compilation, trainer-ready artifacts,
frozen evaluation benchmark, reproducibility manifests, environment files, and
focused tests are complete. The 600-case base benchmark is resumable but was
not allowed to run to completion on the 8 GB laptop: 12 cases were measured
before the sequential runtime budget became disproportionate. No cloud GPU,
training dependency installation, optimizer step, adapter, merge, or
quantization was performed.

The current upstream revision is Qwen/Qwen3.5-9B at
c202236235762e1c871ad0ccb60c8ee5ba337b9a. Its text backbone configuration
matches the production GGUF family, vocabulary, dimensions, layer schedule,
and special-token IDs. The upstream wrapper is multimodal
(Qwen3_5ForConditionalGeneration) while production is text-only; the compiler
uses the exact production GGUF chat template captured from the local server
because the upstream template differs in tool-argument iteration and
thinking-prompt behavior. B4 must run the pinned text-backbone/adapter
round-trip smoke test before training.

Exact tokenization used the pinned tokenizer and production template:
average 426.03, median 430, p90 473, p95 554, p99 615, maximum 635, and zero
records over 1,024 tokens.

Baseline evidence is at data/evaluation/baseline_results.jsonl with summary at
data/evaluation/baseline_summary.json; it contains 12 measured cases and is
resumable with evaluation/run_baseline.py.
