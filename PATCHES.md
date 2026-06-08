# PATCHES

This package vendors official repositories and applies only minimal experiment-compatibility patches.

## Sources

| Baseline | Official source | Vendored commit |
|---|---|---|
| PatchTST | `https://github.com/yuqinie98/PatchTST.git` | `204c21e` |
| DMMV | `https://github.com/D2I-Group/dmmv.git` | `6422971` |
| Time-LLM | `https://github.com/KimMeen/Time-LLM.git` | `b13e881` |
| T3Time | `https://github.com/monaf-chowdhury/T3Time.git` | `6df2d8c` |

## Applied Changes

| Baseline | File | Reason |
|---|---|---|
| DMMV | `Experiment/Exp.py` | Print and save MSE/MAE metrics so the runner can parse official test results. |
| Time-LLM | `run_main.py` | Move random seed setup after argument parsing so `--seed` works for `{2024,2025,2026}`. |
| Time-LLM | `data_provider/data_factory.py` | Add `ILI` and `Exchange` as `Dataset_Custom` keys for the unified nine-dataset protocol. |
| Time-LLM | `dataset/prompt_bank/ILI.txt`, `Exchange.txt` | Add conservative dataset descriptions required by Time-LLM prompt loading. |
| T3Time | `storage/store_emb.py` | Add repo-root import path, configurable data/model paths, embedding batch controls, and `--max_samples` for smoke tests. |
| T3Time | `data_provider/data_loader_save.py` | Replace hardcoded dataset root with `T3TIME_DATA_ROOT`; keep official split logic. |
| T3Time | `data_provider/data_loader_emb.py` | Replace hardcoded embedding root with `T3TIME_EMBED_ROOT`; cache embeddings by input and prediction length. |
| T3Time | `storage/gen_prompt_emb.py` | Allow local GPT-2 path through `T3TIME_GPT2_MODEL_PATH`; add Traffic and Exchange prompt templates. |

## Adapted Combinations

The following combinations are not present as ready-made official shell scripts, but are run through the official custom-data code paths to satisfy the unified experiment protocol:

- DMMV on `Exchange`.
- Time-LLM on `ILI`.
- Time-LLM on `Exchange`.

These should be reported as unified-protocol adaptations, not as paper-table reproductions.
