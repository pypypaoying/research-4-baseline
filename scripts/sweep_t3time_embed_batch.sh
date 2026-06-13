#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-Traffic}"
HORIZON="${HORIZON:-96}"
SEED="${SEED:-2024}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
SPLITS="${SPLITS:-train}"
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-32}"
EMBED_BATCHES="${EMBED_BATCHES:-1 2 4 8 16 32}"
EMBED_WORKERS="${EMBED_WORKERS:-4}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
GPU_TRACE="${GPU_TRACE:-1}"
GPU_SAMPLE_INTERVAL_MS="${GPU_SAMPLE_INTERVAL_MS:-1000}"

RESULT_DIR="results/t3time_embed_batch_sweep_${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
TRACE_DIR="${RESULT_DIR}/gpu_traces"
CACHE_BASE="data/t3time_embed_batch_sweep_${RUN_TAG}"
mkdir -p "$LOG_DIR" "$TRACE_DIR" "$CACHE_BASE" logs results

echo "[python] $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" - <<'PY'
import sys

required = ["numpy", "h5py", "torch", "transformers"]
for name in required:
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "built-in")
        print(f"[env-check] {name}=={version} from {location}")
    except Exception as exc:
        print(f"[env-error] failed to import {name}: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY

COMMON_ARGS=(
  --execute
  --fail-fast
  --t3-embedding-only
  --models T3Time
  --datasets "$DATASET"
  --horizons "$HORIZON"
  --seeds "$SEED"
  --gpu "$GPU"
  --t3-max-embed-samples "$MAX_SAMPLES"
  --t3-embedding-splits "$SPLITS"
  --t3-embed-num-workers "$EMBED_WORKERS"
  --t3-prompt-batch-size "$PROMPT_BATCH_SIZE"
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

start_gpu_trace() {
  local trace_path="$1"
  if [[ "$GPU_TRACE" != "1" ]]; then
    echo ""
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  local smi_args=()
  if [[ "$GPU" =~ ^[0-9]+$ ]]; then
    smi_args=(-i "$GPU")
  fi
  nvidia-smi \
    "${smi_args[@]}" \
    --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu \
    --format=csv \
    -lms "$GPU_SAMPLE_INTERVAL_MS" > "$trace_path" 2>/dev/null &
  echo "$!"
}

stop_gpu_trace() {
  local pid="$1"
  if [[ -n "$pid" ]]; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

run_case() {
  local embed_bs="$1"
  local case_name="embed${embed_bs}_prompt${PROMPT_BATCH_SIZE}"
  local cache_root="${CACHE_BASE}/${case_name}"
  local trace_path="${TRACE_DIR}/${case_name}.csv"
  rm -rf "$cache_root"
  echo
  echo "[case] ${case_name} max_samples=${MAX_SAMPLES} splits=${SPLITS}"
  local trace_pid
  trace_pid="$(start_gpu_trace "$trace_path")"
  set +e
  "$PYTHON_BIN" scripts/run_four_baselines.py \
    "${COMMON_ARGS[@]}" \
    --force-t3-embeddings \
    --t3-embedding-root "$cache_root" \
    --t3-embed-batch-size "$embed_bs"
  local status=$?
  set -e
  stop_gpu_trace "$trace_pid"
  copy_case_logs "$case_name"
  if [[ "$status" -ne 0 ]]; then
    echo "[case-failed] ${case_name} exit=${status}"
    return "$status"
  fi
}

echo "[run-tag] $RUN_TAG"
echo "[config] dataset=$DATASET horizon=$HORIZON seed=$SEED max_samples=$MAX_SAMPLES splits=$SPLITS prompt_batch_size=$PROMPT_BATCH_SIZE"
echo "[sweep] embed_batches=$EMBED_BATCHES"
echo "[cache-base] $CACHE_BASE"

for embed_bs in $EMBED_BATCHES; do
  if ! run_case "$embed_bs"; then
    echo "[stop] embed_batch=${embed_bs} failed; sweep stops here."
    break
  fi
done

"$PYTHON_BIN" scripts/summarize_t3time_embedding_benchmark.py \
  --input-dir "$LOG_DIR" \
  --output "${RESULT_DIR}/summary.csv"

cat > "${RESULT_DIR}/README.txt" <<EOF
T3Time embedding-batch sweep

This benchmark only runs storage/store_emb.py through scripts/run_four_baselines.py
with --t3-embedding-only. It does not train T3Time and it does not test cache-hit reuse.

Dataset: ${DATASET}
Horizon: ${HORIZON}
Seed: ${SEED}
Max samples per split: ${MAX_SAMPLES}
Splits: ${SPLITS}
Prompt batch size: ${PROMPT_BATCH_SIZE}
Embedding batches: ${EMBED_BATCHES}
GPU trace enabled: ${GPU_TRACE}
Cache base: ${CACHE_BASE}

Each embedding batch size uses a fresh cache root and --force-t3-embeddings.
Compare seconds, samples_per_min, prompts_per_sec_est, peak_mem_ratio, and oom.
Summary CSV: ${RESULT_DIR}/summary.csv
EOF

echo
echo "[saved] ${RESULT_DIR}/summary.csv"
echo "[saved] ${RESULT_DIR}/README.txt"
