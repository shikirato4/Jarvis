#!/usr/bin/env bash
set -euo pipefail

# B3 artifact only. This script is intentionally not executed in B3.
# It expects a sanitized bundle containing compiled train/validation JSONL,
# pilot_training.json, requirements-linux-cu130.txt, and the training entrypoint.
# It must never read .env, credentials, memory databases, held-out data, or
# personal files.

MODEL_ID="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONFIG="configs/pilot_training.json"
DATA_ROOT="data/compiled"

echo "B4-only training command template for $MODEL_ID@$MODEL_REVISION"
echo "Install the locked requirements, verify hashes, then invoke the reviewed trainer."
echo "Inputs: $DATA_ROOT/train.jsonl $DATA_ROOT/validation.jsonl $CONFIG"
echo "Outputs: adapter-only checkpoint, metrics, and reproducibility manifest"
