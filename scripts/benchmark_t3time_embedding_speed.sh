#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-Traffic}"
HORIZON="${HORIZON:-96}"
SEED="${SEED:-2024}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
SPLITS="${SPLITS:-train}"
EMBED_WORKERS="${EMBED_WORKERS:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Keep defaults small enough for a quick 3090 check. Increase these on A800 after
# the trend is clear.
PROMPT_BATCHES="${PROMPT_BATCHES:-1 8 32}"
SAMPLE_BATCHES="${SAMPLE_BATCHES:-1 2}"
CACHE_PROMPT_BATCH="${CACHE_PROMPT_BATCH:-32}"
CACHE_SAMPLE_BATCH="${CACHE_SAMPLE_BATCH:-1}"

RESULT_DIR="results/t3time_embedding_benchmark_${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
CACHE_BASE="data/t3time_embedding_benchmark_${RUN_TAG}"
mkdir -p "$LOG_DIR" "$CACHE_BASE" logs results

COMMON_ARGS=(
  --execute
  --t3-embedding-only
  --models T3Time
  --datasets "$DATASET"
  --horizons "$HORIZON"
  --seeds "$SEED"
  --gpu "$GPU"
  --t3-max-embed-samples "$MAX_SAMPLES"
  --t3-embedding-splits "$SPLITS"
  --t3-embed-num-workers "$EMBED_WORKERS"
)

if [[ -n "${T3TIME_PYTHON:-}" ]]; then
  COMMON_ARGS+=(--t3time-python "$T3TIME_PYTHON")
fi
if [[ -n "${T3TIME_GPT2_MODEL_PATH:-}" ]]; then
  COMMON_ARGS+=(--t3-gpt2-model-path "$T3TIME_GPT2_MODEL_PATH")
fi
if [[ "${T3TIME_GPT2_LOCAL_ONLY:-0}" == "1" ]]; then
  COMMON_ARGS+=(--t3-gpt2-local-only)
fi

copy_case_logs() {
  local case_name="$1"
  local case_dir="${LOG_DIR}/${case_name}"
  mkdir -p "$case_dir"
  local split
  IFS=',' read -ra split_array <<< "$SPLITS"
  for split in "${split_array[@]}"; do
    local src="logs/T3Time_embed_${DATASET}_h${HORIZON}_${split}_seed${SEED}.log"
    if [[ -f "$src" ]]; then
      cp "$src" "${case_dir}/${DATASET}_${split}.log"
    fi
  done
}

run_case() {
  local case_name="$1"
  local embed_bs="$2"
  local prompt_bs="$3"
  local cache_root="${CACHE_BASE}/${case_name}"
  rm -rf "$cache_root"
  echo
  echo "[case] ${case_name} embed_batch=${embed_bs} prompt_batch=${prompt_bs}"
  "$PYTHON_BIN" scripts/run_four_baselines.py \
    "${COMMON_ARGS[@]}" \
    --force-t3-embeddings \
    --t3-embedding-root "$cache_root" \
    --t3-embed-batch-size "$embed_bs" \
    --t3-prompt-batch-size "$prompt_bs"
  copy_case_logs "$case_name"
}

echo "[run-tag] $RUN_TAG"
echo "[config] dataset=$DATASET horizon=$HORIZON seed=$SEED max_samples=$MAX_SAMPLES splits=$SPLITS"
echo "[cache-base] $CACHE_BASE"

run_case "official_prompt1_sample1" 1 1

for prompt_bs in $PROMPT_BATCHES; do
  if [[ "$prompt_bs" == "1" ]]; then
    continue
  fi
  run_case "prompt${prompt_bs}_sample1" 1 "$prompt_bs"
done

for embed_bs in $SAMPLE_BATCHES; do
  if [[ "$embed_bs" == "1" ]]; then
    continue
  fi
  run_case "prompt${CACHE_PROMPT_BATCH}_sample${embed_bs}" "$embed_bs" "$CACHE_PROMPT_BATCH"
done

CACHE_CASE="cache_reuse_prompt${CACHE_PROMPT_BATCH}_sample${CACHE_SAMPLE_BATCH}"
CACHE_ROOT="${CACHE_BASE}/${CACHE_CASE}"
rm -rf "$CACHE_ROOT"
echo
echo "[case] ${CACHE_CASE}_fill"
"$PYTHON_BIN" scripts/run_four_baselines.py \
  "${COMMON_ARGS[@]}" \
  --force-t3-embeddings \
  --t3-embedding-root "$CACHE_ROOT" \
  --t3-embed-batch-size "$CACHE_SAMPLE_BATCH" \
  --t3-prompt-batch-size "$CACHE_PROMPT_BATCH"
copy_case_logs "${CACHE_CASE}_fill"

echo
echo "[case] ${CACHE_CASE}_hit"
"$PYTHON_BIN" scripts/run_four_baselines.py \
  "${COMMON_ARGS[@]}" \
  --t3-run-cache-checks \
  --t3-embedding-root "$CACHE_ROOT" \
  --t3-embed-batch-size "$CACHE_SAMPLE_BATCH" \
  --t3-prompt-batch-size "$CACHE_PROMPT_BATCH"
copy_case_logs "${CACHE_CASE}_hit"

"$PYTHON_BIN" scripts/summarize_t3time_embedding_benchmark.py \
  --input-dir "$LOG_DIR" \
  --output "${RESULT_DIR}/summary.csv"

cat > "${RESULT_DIR}/README.txt" <<EOF
T3Time embedding speed benchmark

This benchmark only runs storage/store_emb.py through scripts/run_four_baselines.py
with --t3-embedding-only. It does not train T3Time.

Dataset: ${DATASET}
Horizon: ${HORIZON}
Seed: ${SEED}
Max samples per split: ${MAX_SAMPLES}
Splits: ${SPLITS}
Cache base: ${CACHE_BASE}

Fairness-preserving speed levers tested:
- prompt batching: same prompts, same GPT-2 model, same FP32 path, same last-token embedding;
- sample batching: same sample-level output files, generated in larger input batches;
- cache reuse: embeddings are keyed by input length because the prompt branch uses x and x_mark, not pred_len labels.

Summary CSV: ${RESULT_DIR}/summary.csv
EOF

echo
echo "[saved] ${RESULT_DIR}/summary.csv"
echo "[saved] ${RESULT_DIR}/README.txt"
