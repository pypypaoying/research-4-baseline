#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-2024}"
HORIZON="${HORIZON:-96}"
THRESHOLD="${MEMORY_ANOMALY_THRESHOLD:-0.90}"

PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")/.."

echo "[probe] DMMV Traffic h=${HORIZON} seed=${SEED} gpu=${GPU} batch_size=8"
"${PYTHON_BIN}" scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models DMMV \
  --datasets Traffic \
  --horizons "${HORIZON}" \
  --seeds "${SEED}" \
  --gpu "${GPU}" \
  --memory-anomaly-threshold "${THRESHOLD}"

echo "[probe] TimeLLM Traffic h=${HORIZON} seed=${SEED} gpu=${GPU} batch_size=32 GPT2 layers=6 fp32"
"${PYTHON_BIN}" scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models TimeLLM \
  --datasets Traffic \
  --horizons "${HORIZON}" \
  --seeds "${SEED}" \
  --gpu "${GPU}" \
  --batch-size 32 \
  --timellm-llm-model GPT2 \
  --timellm-llm-layers 6 \
  --timellm-processes 1 \
  --timellm-mixed-precision no \
  --memory-anomaly-threshold "${THRESHOLD}"

"${PYTHON_BIN}" scripts/summarize_memory_probe.py
