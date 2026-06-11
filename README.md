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
  --timellm-mixed-precision no \
  --fail-fast
```

Useful probe controls:

- `--probe-epochs 1`: number of epochs for memory probing.
- `--probe-max-train-batches 2`: T3Time-only train batch cap during probe.
- `--probe-max-eval-batches 1`: T3Time-only eval/test batch cap during probe.
- `--probe-t3-max-embed-samples 8`: T3Time embedding samples during probe.
- `--gpu-sample-interval-ms 500`: `nvidia-smi` sampling interval.
- `--memory-anomaly-threshold 0.90`: mark memory anomaly if peak memory reaches 90% of visible GPU memory.

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
