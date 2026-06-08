# STRUCTURE

```text
research-4-baseline/
  baselines/
    PatchTST/      # vendored official repo, source commit 204c21e
    dmmv/          # vendored official repo, source commit 6422971
    Time-LLM/      # vendored official repo, source commit b13e881
    T3Time/        # vendored official repo, source commit 6df2d8c
  configs/
    experiment1_fullshot.yaml
  data/            # ignored; shared benchmark CSVs and T3Time embeddings
  checkpoints/     # ignored; DMMV MAE checkpoint and model caches if needed
  logs/            # ignored; raw run logs
  results/         # ignored; runs.csv and summary CSVs
  scripts/
    prepare_data_layout.py
    download_datasets.py
    download_checkpoints.py
    run_four_baselines.py
    summarize_results.py
```

Canonical experiment data lives under `data/`; do not use vendor-local `dataset/` folders for this experiment.

T3Time uses two special shared paths:

- `data/t3time_dataset/`: CSV aliases named as T3Time expects, such as `Traffic.csv`.
- `data/t3time_embeddings/<dataset>/seq336_pred<horizon>/<split>/`: cached prompt embeddings.

The runner calls official Python entrypoints, not the vendor shell scripts.
