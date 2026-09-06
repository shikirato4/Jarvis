# JARVIS Agentic LoRA dataset tooling

This directory is isolated from src/jarvis production runtime code. It builds
the canonical jarvis.agentic_lora.v1 JSONL pilot from deterministic,
sanitized scenarios.

## Build

~~~text
python -m training.agentic_lora.build_pilot
~~~

The command writes data/canonical/{train,validation,heldout,hard_regression}.jsonl,
data/dataset_report.json, and data/dataset_report.md. It does not download
models, install packages, contact an LLM, or run tools.

Current real capabilities are sanitized registry-shaped snapshots in
real_tool_snapshots.py. B3 may replace these with a hash-pinned export from
ActionExecutor.tool_schemas(); no permanent production tool catalog is created.
