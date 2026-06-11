# PLAN

## Goal

Build a lightweight GitHub-managed package for the four problematic official baselines: PatchTST, DMMV, Time-LLM, and T3Time. The package should be cloned on the 2x3090 server and run the unified full-shot protocol without depending on the previous cloud wrapper.

## Experiment Contract

- Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, ILI, Exchange.
- Splits: ETT loaders use fixed 12/4/4 month splits; custom datasets use chronological 70/10/20.
- Input length: 336.
- Horizons: 96, 192, 336, 720, except ILI uses 24, 36, 48, 60.
- Metrics: MSE and MAE.
- Seeds: 2024, 2025, 2026.
- Training: 15 epochs with patience 5.
- Summary: per dataset-horizon MSE/MAE, dataset average, horizon average, overall rank, and first count.

## Route

Use vendored official repositories with minimal compatibility patches:

- PatchTST: official `PatchTST_supervised/run_longExp.py`.
- DMMV: official `run.py`; patch only metric printing/saving if needed.
- Time-LLM: official `run_main.py`; patch seed handling so `--seed` actually controls randomness, and add custom dataset keys for ILI/Exchange.
- T3Time: official `train.py` and `storage/store_emb.py`; patch hardcoded data/embedding/model paths into environment-controlled paths.

## Expected Outputs

- `results/runs.csv`: one row per successful run.
- `logs/*.log`: raw official stdout/stderr for each run.
- `results/summary/*.csv`: seed-averaged case table, dataset averages, horizon averages, overall ranks, and first counts.
- `results/memory_probe.csv`: short-run memory probe records with OOM and peak GPU memory.
- `results/gpu_traces/*.csv`: raw `nvidia-smi` traces for memory probe runs.

## Risks

- Time-LLM with LLaMA is heavy; use `--timellm-llm-model GPT2` only for smoke or resource triage.
- T3Time embedding generation is expensive, especially for Traffic. Embeddings are cached by dataset, input length, prediction length, and split.
- DMMV requires the MAE checkpoint at `checkpoints/mae_visualize_vit_base.pth`.
- Exchange for DMMV and ILI/Exchange for Time-LLM are unified-protocol adaptations rather than official shell-script combinations.
