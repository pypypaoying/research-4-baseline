#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-2024}"
DATASETS="${DATASETS:-ETTh1,ETTh2,ETTm1,ETTm2,Weather,ECL,Traffic,ILI,Exchange}"
HORIZONS="${HORIZONS:-96,192,336,720}"
THRESHOLD="${MEMORY_ANOMALY_THRESHOLD:-0.90}"
TIMELLM_ACCELERATE="${TIMELLM_ACCELERATE:-accelerate}"
TIMELLM_PROCESSES="${TIMELLM_PROCESSES:-1}"
TIMELLM_BATCH_SIZE="${TIMELLM_BATCH_SIZE:-24}"
MASTER_PORT="${MASTER_PORT:-29024}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$(dirname "$0")/.."

echo "[probe] TimeLLM official-like config"
echo "[probe] datasets=${DATASETS}"
echo "[probe] horizons=${HORIZONS}"
echo "[probe] seed=${SEED} gpu=${GPU}"
echo "[probe] seq_len=336 batch_size=${TIMELLM_BATCH_SIZE} llm_model=LLAMA llm_layers=32 mixed_precision=bf16 processes=${TIMELLM_PROCESSES}"

"${PYTHON_BIN}" scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models TimeLLM \
  --datasets "${DATASETS}" \
  --horizons "${HORIZONS}" \
  --seeds "${SEED}" \
  --gpu "${GPU}" \
  --batch-size "${TIMELLM_BATCH_SIZE}" \
  --timellm-llm-model LLAMA \
  --timellm-llm-layers 32 \
  --timellm-processes "${TIMELLM_PROCESSES}" \
  --timellm-mixed-precision bf16 \
  --timellm-accelerate "${TIMELLM_ACCELERATE}" \
  --master-port "${MASTER_PORT}" \
  --probe-epochs 1 \
  --memory-anomaly-threshold "${THRESHOLD}"

"${PYTHON_BIN}" scripts/summarize_memory_probe.py
