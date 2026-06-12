#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
FRESH_CACHE="${FRESH_CACHE:-0}"
KEEP_OLD_RUNS="${KEEP_OLD_RUNS:-0}"

# Official-style preprocessing: one dataloader sample at a time, one GPT-2 prompt at a time.
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-1}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-1}"

CACHE_DIR="data/t3time_embeddings/Traffic/seq336"
LOG_FILE="logs/t3time_traffic_official_style_${RUN_TAG}.log"
SUMMARY_DIR="results/summary_t3time_traffic_official_style_${RUN_TAG}"

mkdir -p logs results

if [[ "$FRESH_CACHE" == "1" && -d "$CACHE_DIR" ]]; then
  mv "$CACHE_DIR" "${CACHE_DIR}.bak_${RUN_TAG}"
  echo "[cache] moved $CACHE_DIR to ${CACHE_DIR}.bak_${RUN_TAG}"
fi

if [[ "$KEEP_OLD_RUNS" != "1" && -f "results/runs.csv" ]]; then
  mv "results/runs.csv" "results/runs.before_t3time_traffic_${RUN_TAG}.csv"
  echo "[results] moved previous results/runs.csv to results/runs.before_t3time_traffic_${RUN_TAG}.csv"
fi

{
  echo "[run-tag] $RUN_TAG"
  echo "[start] $(date -Is)"
  echo "[config] dataset=Traffic split=70/10/20 seq_len=336 horizons=96,192,336,720 seed=2024"
  echo "[official-style] embed_batch_size=${EMBED_BATCH_SIZE} prompt_batch_size=${PROMPT_BATCH_SIZE}"
  echo "[fresh-cache] ${FRESH_CACHE}"

  start_ts=$(date +%s)

  "$PYTHON_BIN" scripts/run_four_baselines.py \
    --execute \
    --models T3Time \
    --datasets Traffic \
    --horizons 96,192,336,720 \
    --seeds 2024 \
    --gpu "$GPU" \
    --t3-embed-batch-size "$EMBED_BATCH_SIZE" \
    --t3-prompt-batch-size "$PROMPT_BATCH_SIZE"

  if [[ -f "results/runs.csv" ]]; then
    cp "results/runs.csv" "results/t3time_traffic_official_style_${RUN_TAG}.csv"
    "$PYTHON_BIN" scripts/summarize_results.py \
      --input "results/t3time_traffic_official_style_${RUN_TAG}.csv" \
      --output-dir "$SUMMARY_DIR"
  fi

  end_ts=$(date +%s)
  echo "[end] $(date -Is)"
  echo "[elapsed_seconds] $((end_ts - start_ts))"
  echo "[log] $LOG_FILE"
  echo "[summary] $SUMMARY_DIR"
} 2>&1 | tee "$LOG_FILE"
