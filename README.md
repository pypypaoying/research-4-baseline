# research-4-baseline

This repository packages four official baseline codebases for the unified full-shot long-term forecasting experiment:

- `PatchTST`
- `DMMV`
- `Time-LLM`
- `T3Time`

The code is vendored under `baselines/` so the server only needs one Git clone. The runner is a thin wrapper that launches each official training entrypoint with the shared experiment settings.

## Protocol

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `ECL`, `Traffic`, `ILI`, `Exchange`
- Splits: ETT uses the official 12/4/4 month split; other datasets use chronological `70/10/20`
- Input length: `336`
- Horizons: `{96,192,336,720}`, except ILI uses `{24,36,48,60}`
- Metrics: `MSE`, `MAE`
- Seeds: `{2024,2025,2026}`
- Training: `15` epochs, early stopping patience `5`

## Server Setup

```bash
git clone https://github.com/pypypaoying/research-4-baseline.git
cd research-4-baseline

conda create -n r4b python=3.10 -y
conda activate r4b
pip install pyyaml pandas huggingface_hub
pip install -r baselines/Time-LLM/requirements.txt
pip install -r baselines/dmmv/requirements.txt
pip install -r baselines/PatchTST/PatchTST_supervised/requirements.txt
pip install h5py timm tensorboard
```

If dependency conflicts appear, use separate conda envs per model. The runner itself is model-agnostic, but the official repositories have different dependency ages.

When using separate envs, pass executables explicitly:

```bash
python scripts/run_four_baselines.py --execute \
  --patchtst-python /path/to/patchtst/bin/python \
  --dmmv-python /path/to/dmmv/bin/python \
  --t3time-python /path/to/t3time/bin/python \
  --timellm-accelerate /path/to/timellm/bin/accelerate
```

## Data

Option A, download from HuggingFace:

```bash
python scripts/download_datasets.py --prepare-t3time
```

Option B, use an existing local dataset root:

```bash
python scripts/prepare_data_layout.py --source-root /path/to/datasets --mode symlink
```

The expected source root layout is:

```text
/path/to/datasets/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  weather/weather.csv
  electricity/electricity.csv
  traffic/traffic.csv
  illness/national_illness.csv
  exchange_rate/exchange_rate.csv
```

## Checkpoints

DMMV needs the MAE checkpoint:

```bash
python scripts/download_checkpoints.py
```

Time-LLM defaults to the official LLaMA setting. For quick smoke tests, use GPT-2:

```bash
--timellm-llm-model GPT2 --timellm-llm-layers 6 --timellm-processes 1 --timellm-mixed-precision no
```

T3Time uses GPT-2 for prompt embeddings. To force an offline/local cache:

```bash
--t3-gpt2-model-path /path/to/gpt2 --t3-gpt2-local-only
```

For Traffic, T3Time's prompt embedding stage is the main bottleneck: every time-series window has 862 variables. The official repository provides offline `Store_{data_name}.sh` scripts and `storage/store_emb.py` / `storage/gen_prompt_emb.py` logic that store GPT-2 last-token embeddings before training; in the original code path, prompt embeddings are produced per variable, so this cost is amplified on high-dimensional datasets such as Traffic. This package keeps the same prompts, GPT-2 model, FP32 inference, and last-token embedding contract, but removes avoidable engineering overhead in two ways:

- `--t3-prompt-batch-size` batches prompt inference without changing the generated prompts.
- T3Time embeddings are cached by input length, for example `data/t3time_embeddings/Traffic/seq336/train/`, because the prompt embedding depends on the input window and timestamp, not on `pred_len`. The same cache is reused across horizons and seeds; if a shorter horizon needs more tail windows, `store_emb.py` only fills the missing files.

To diagnose only this preprocessing stage on a 3090 before running full training:

```bash
python scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --t3-embedding-only \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --t3-max-embed-samples 8 \
  --t3-prompt-batch-size 32
```

This writes limited probe embeddings under `data/t3time_embeddings_probe/` and memory traces under `results/gpu_traces/`. It does not populate the full-run cache, so a later A800 full run will still generate the complete embeddings under `data/t3time_embeddings/`.

To focus on the original slow-preprocessing issue rather than memory, run the dedicated speed benchmark:

```bash
MAX_SAMPLES=8 SPLITS=train GPU=0 bash scripts/benchmark_t3time_embedding_speed.sh
```

The benchmark does not train T3Time. It compares official-style preprocessing (`prompt_batch_size=1`, `embedding_batch_size=1`) with prompt batching, sample batching, and a cache-hit pass. The summary is written to `results/t3time_embedding_benchmark_<timestamp>/summary.csv`. On A800, increase `MAX_SAMPLES`, for example `MAX_SAMPLES=64`, after the 8-sample trend is clear.

Useful knobs:

```bash
MAX_SAMPLES=16 SPLITS=train,val,test PROMPT_BATCHES="8 32 64" SAMPLE_BATCHES="1 2 4" \
GPU=0 bash scripts/benchmark_t3time_embedding_speed.sh
```

These knobs preserve the baseline meaning when they keep the same GPT-2 model, FP32 inference, prompt text, and last-token embedding contract. They only change how many prompts or samples are processed per forward pass and whether already-generated embeddings are reused.

For full T3Time runs, generate the shortest-horizon cache first to cover the largest number of input windows, then later horizons and seeds will reuse it:

```bash
python scripts/run_four_baselines.py \
  --execute \
  --models T3Time \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --t3-embed-batch-size 4 \
  --t3-prompt-batch-size 128
```

After that, running `--horizons 192,336,720` or more seeds should only do a fast cache check for embeddings unless files are missing.

To measure the slow official-style T3Time Traffic path under the unified protocol, run one GPT-2 prompt at a time and one embedding sample at a time:

```bash
FRESH_CACHE=1 GPU=0 bash scripts/run_t3time_traffic_official_style.sh
```

This runs Traffic, `seq_len=336`, horizons `96,192,336,720`, seed `2024`, and summarizes results under `results/summary_t3time_traffic_official_style_<timestamp>/`. `FRESH_CACHE=1` moves any existing `data/t3time_embeddings/Traffic/seq336/` cache aside before starting, so the wall-clock time includes full embedding generation.

## Dry Run

```bash
python scripts/run_four_baselines.py \
  --dry-run \
  --models PatchTST,DMMV,TimeLLM,T3Time \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024
```

## Smoke Run

```bash
python scripts/run_four_baselines.py \
  --execute \
  --models PatchTST,DMMV,TimeLLM,T3Time \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --timellm-llm-model GPT2 \
  --timellm-llm-layers 6 \
  --timellm-processes 1 \
  --timellm-mixed-precision no \
  --t3-max-embed-samples 64 \
  --fail-fast
```

## Memory Probe

Use this when the goal is to record whether the four problematic models have abnormal GPU memory behavior. It short-runs the official commands, samples `nvidia-smi`, writes raw traces, and records OOM/peak memory in `results/memory_probe.csv`.
This is the recommended first step for diagnosing the four excluded/problematic baselines; it is not intended to produce final forecasting scores.

```bash
python scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models PatchTST,DMMV,TimeLLM,T3Time \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --timellm-llm-model GPT2 \
  --timellm-llm-layers 6 \
  --timellm-processes 1 \
  --timellm-mixed-precision no
```

In this all-model command, DMMV uses the configured Traffic `batch_size=8`; Time-LLM uses the configured default `batch_size=24` unless `--batch-size` is supplied.

For the two remaining high-memory cases, run them separately so the Time-LLM batch-size override does not accidentally overwrite DMMV's configured Traffic batch size:

```bash
GPU=0 bash scripts/run_dmmv_timellm_traffic_probe.sh
```

This fixed probe uses:

- DMMV: Traffic, horizon 96, seed 2024, `seq_len=336`, `batch_size=8`.
- Time-LLM: Traffic, horizon 96, seed 2024, `seq_len=336`, `batch_size=32`, GPT-2 with 6 layers, one process, FP32/no mixed precision.

For DMMV memory probes, the runner caps the DMMV train/eval/test loops with `DMMV_MAX_TRAIN_BATCHES` and `DMMV_MAX_EVAL_BATCHES`; this avoids accidentally running a full Traffic epoch just to test memory. Full non-probe runs do not set these limits.

To try a lower DMMV rescue batch size:

```bash
DMMV_BATCH_SIZE=2 GPU=0 bash scripts/run_dmmv_timellm_traffic_probe.sh
```

Do not use `--fail-fast` for the first memory sweep. If one model OOMs, that is a valid probe result and the runner should continue to the remaining models.

If a model OOMs under the official-like batch size, keep that row as the primary memory finding. Then run a second, clearly marked rescue probe with a smaller batch size to check whether the issue is batch-size driven:

```bash
python scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models PatchTST \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --batch-size 8
```

Useful probe controls:

- `--probe-epochs 1`: number of epochs for memory probing.
- `--probe-max-train-batches 2`: T3Time-only train batch cap during probe.
- `--probe-max-eval-batches 1`: T3Time-only eval/test batch cap during probe.
- `--probe-t3-max-embed-samples 8`: T3Time embedding samples during probe.
- `--t3-embedding-only`: run only T3Time embedding generation and skip `train.py`.
- `--t3-prompt-batch-size 32`: batch GPT-2 prompt inference during T3Time embedding generation.
- `--gpu-sample-interval-ms 500`: `nvidia-smi` sampling interval.
- `--memory-anomaly-threshold 0.90`: mark memory anomaly if peak memory reaches 90% of visible GPU memory.

To test the exact T3Time training memory at batch size 16 after a small embedding probe cache has been created:

```bash
python scripts/run_four_baselines.py \
  --execute \
  --memory-probe \
  --models T3Time \
  --datasets Traffic \
  --horizons 96 \
  --seeds 2024 \
  --gpu 0 \
  --batch-size 16
```

Summarize memory probe results:

```bash
python scripts/summarize_memory_probe.py
```

Outputs:

- `results/memory_probe.csv`
- `results/gpu_traces/*.csv`
- `results/summary/memory_probe_case_summary.csv`
- `results/summary/memory_probe_model_summary.csv`

## Full Run

```bash
python scripts/run_four_baselines.py --execute
```

Run one process per GPU by filtering model/dataset groups manually, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_four_baselines.py --execute --models PatchTST,DMMV --gpu 0
CUDA_VISIBLE_DEVICES=1 python scripts/run_four_baselines.py --execute --models T3Time --gpu 0
```

For Time-LLM with the default LLaMA setting, leave both 3090 cards visible or set `--timellm-processes 1` for a single visible GPU.

## Summary

```bash
python scripts/summarize_results.py
```

Outputs are written to `results/summary/`:

- `per_dataset_horizon_seedavg.csv`
- `per_dataset_horizon_ranked.csv`
- `dataset_average.csv`
- `horizon_average.csv`
- `overall_rank_first_count.csv`

## Notes

- Do not run the old vendor shell scripts directly unless you are reproducing a paper-specific command. Some vendor scripts contain personal paths or old epoch settings.
- The main runner keeps official model hyperparameters where available and only overrides the unified protocol fields.
- Exchange for DMMV and ILI/Exchange for Time-LLM are unified-protocol adaptations through existing custom-data loaders.
- See `PATCHES.md` for source commits and exact compatibility changes.
