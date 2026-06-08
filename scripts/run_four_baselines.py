#!/usr/bin/env python3
"""Run the four official baselines under the unified full-shot protocol."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = REPO_ROOT / "results" / "runs.csv"
MODEL_ORDER = ["PatchTST", "DMMV", "TimeLLM", "T3Time"]


@dataclass(frozen=True)
class RunSpec:
    model: str
    dataset: str
    horizon: int
    seed: int


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_arg(value: str | None, default: Iterable[str]) -> list[str]:
    if not value:
        return list(default)
    return [v.strip() for v in value.split(",") if v.strip()]


def str_path(path: Path) -> str:
    return str(path.resolve())


def mkdirs() -> None:
    for rel in ["logs", "results", "checkpoints", "data/t3time_dataset"]:
        (REPO_ROOT / rel).mkdir(parents=True, exist_ok=True)


def python_for(model: str, args: argparse.Namespace) -> str:
    attr = {
        "PatchTST": "patchtst_python",
        "DMMV": "dmmv_python",
        "T3Time": "t3time_python",
    }.get(model)
    cli_value = getattr(args, attr, None) if attr else None
    env_value = os.environ.get(f"{model.upper().replace('-', '_')}_PYTHON")
    return cli_value or env_value or args.python or sys.executable


def env_with_gpu(gpu: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if extra:
        env.update(extra)
    return env


def bool_flag(value: bool) -> str:
    return "1" if value else "0"


def patchtst_hparams(config: dict, dataset: str, horizon: int) -> dict:
    spec = config["datasets"][dataset]
    mcfg = config["models"]["PatchTST"]
    is_ett_hour = dataset in {"ETTh1", "ETTh2"}
    is_ili = dataset == "ILI"
    hp = {
        "batch_size": mcfg.get("dataset_batch_size", {}).get(dataset, mcfg["default_batch_size"]),
        "learning_rate": mcfg.get("dataset_learning_rate", {}).get(dataset, mcfg["learning_rate"]),
        "lradj": mcfg.get("dataset_lradj", {}).get(dataset, mcfg["lradj"]),
        "patch_len": mcfg.get("dataset_patch_len", {}).get(dataset, mcfg["patch_len"]),
        "stride": mcfg.get("dataset_stride", {}).get(dataset, mcfg["stride"]),
        "e_layers": 3,
        "n_heads": 4 if is_ett_hour or is_ili else 16,
        "d_model": 16 if is_ett_hour or is_ili else 128,
        "d_ff": 128 if is_ett_hour or is_ili else 256,
        "dropout": 0.3 if is_ett_hour or is_ili else 0.2,
        "fc_dropout": 0.3 if is_ett_hour or is_ili else 0.2,
        "head_dropout": 0,
        "pct_start": 0.2,
    }
    if dataset in {"ETTm1", "ETTm2"}:
        hp["pct_start"] = 0.4
    if dataset == "ILI":
        hp["pct_start"] = 0.3
    if dataset == "Exchange":
        hp.update(
            {
                "n_heads": 4,
                "d_model": 16,
                "d_ff": 128,
                "dropout": 0.3,
                "fc_dropout": 0.3,
                "batch_size": 8 if horizon in {96, 192} else 32,
            }
        )
    hp["enc_in"] = spec["n_vars"]
    return hp


def dmmv_hparams(config: dict, dataset: str) -> dict:
    mcfg = config["models"]["DMMV"]
    return {
        "batch_size": mcfg.get("dataset_batch_size", {}).get(dataset, mcfg["default_batch_size"]),
        "optimizer": mcfg["optimizer"],
        "learning_rate": mcfg["learning_rate"],
        "individual": mcfg["individual"],
        "norm_const": mcfg["norm_const"],
        "align_const": mcfg["align_const"],
        "vm_arch": mcfg["vm_arch"],
        "ft_type": mcfg["ft_type"],
        "model": mcfg["variant"],
    }


def timellm_hparams(config: dict, dataset: str, overrides: argparse.Namespace) -> dict:
    mcfg = config["models"]["TimeLLM"]
    llm_model = overrides.timellm_llm_model or mcfg["llm_model"]
    if llm_model.upper() == "GPT2":
        llm_dim = 768
    elif llm_model.upper() == "BERT":
        llm_dim = 768
    else:
        llm_dim = mcfg["llm_dim"]
    return {
        "batch_size": overrides.batch_size or mcfg["default_batch_size"],
        "learning_rate": mcfg["learning_rate"],
        "d_model": mcfg.get("dataset_d_model", {}).get(dataset, mcfg["d_model"]),
        "d_ff": mcfg.get("dataset_d_ff", {}).get(dataset, mcfg["d_ff"]),
        "llm_model": llm_model.upper(),
        "llm_dim": llm_dim,
        "llm_layers": overrides.timellm_llm_layers or mcfg["llm_layers"],
        "processes": overrides.timellm_processes or mcfg["accelerate_processes"],
        "mixed_precision": overrides.timellm_mixed_precision or mcfg["mixed_precision"],
    }


def t3_hparams(config: dict, dataset: str, horizon: int, overrides: argparse.Namespace) -> dict:
    mcfg = config["models"]["T3Time"]
    by_ds = mcfg.get("official_like_by_dataset_horizon", {})
    hp = dict(by_ds.get(dataset, {}).get(str(horizon), {}))
    hp.setdefault("batch_size", mcfg["default_batch_size"])
    hp.setdefault("channel", mcfg["default_channel"])
    hp.setdefault("learning_rate", mcfg["default_learning_rate"])
    hp.setdefault("dropout_n", mcfg["default_dropout_n"])
    hp.setdefault("e_layer", mcfg["default_e_layer"])
    hp.setdefault("d_layer", mcfg["default_d_layer"])
    hp.setdefault("num_workers", mcfg.get("dataset_num_workers", {}).get(dataset, 10))
    if overrides.batch_size:
        hp["batch_size"] = overrides.batch_size
    return hp


def command_for(spec: RunSpec, config: dict, args: argparse.Namespace) -> tuple[list[str], Path, Path, dict[str, str]]:
    ds = config["datasets"][spec.dataset]
    seq_len = int(config["seq_len"])
    label_len = int(config["label_len"])
    epochs = int(config["epochs"])
    patience = int(config["patience"])
    root_path = REPO_ROOT / ds["root"]
    log_path = REPO_ROOT / "logs" / f"{spec.model}_{spec.dataset}_h{spec.horizon}_seed{spec.seed}.log"

    if spec.model == "PatchTST":
        hp = patchtst_hparams(config, spec.dataset, spec.horizon)
        cwd = REPO_ROOT / "baselines" / "PatchTST" / "PatchTST_supervised"
        cmd = [
            python_for("PatchTST", args),
            "-u",
            "run_longExp.py",
            "--random_seed",
            str(spec.seed),
            "--is_training",
            "1",
            "--root_path",
            str_path(root_path),
            "--data_path",
            ds["file"],
            "--model_id",
            f"{spec.dataset}_{seq_len}_{spec.horizon}_seed{spec.seed}",
            "--model",
            "PatchTST",
            "--data",
            ds["data"],
            "--features",
            "M",
            "--seq_len",
            str(seq_len),
            "--label_len",
            str(label_len),
            "--pred_len",
            str(spec.horizon),
            "--enc_in",
            str(hp["enc_in"]),
            "--e_layers",
            str(hp["e_layers"]),
            "--n_heads",
            str(hp["n_heads"]),
            "--d_model",
            str(hp["d_model"]),
            "--d_ff",
            str(hp["d_ff"]),
            "--dropout",
            str(hp["dropout"]),
            "--fc_dropout",
            str(hp["fc_dropout"]),
            "--head_dropout",
            str(hp["head_dropout"]),
            "--patch_len",
            str(hp["patch_len"]),
            "--stride",
            str(hp["stride"]),
            "--des",
            "Exp1",
            "--train_epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--lradj",
            hp["lradj"],
            "--pct_start",
            str(hp["pct_start"]),
            "--itr",
            "1",
            "--batch_size",
            str(args.batch_size or hp["batch_size"]),
            "--learning_rate",
            str(hp["learning_rate"]),
        ]
        return cmd, cwd, log_path, {}

    if spec.model == "DMMV":
        hp = dmmv_hparams(config, spec.dataset)
        mcfg = config["models"]["DMMV"]
        cwd = REPO_ROOT / "baselines" / "dmmv"
        ckpt = REPO_ROOT / mcfg["trained_MAE_ckpt"]
        data_name = ds["data"] if ds["data"] in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"} else "custom"
        cmd = [
            python_for("DMMV", args),
            "-u",
            "run.py",
            "--exp_info",
            f"DMMV_A_{spec.dataset}_{seq_len}_{spec.horizon}_seed{spec.seed}",
            "--model",
            hp["model"],
            "--random_seed",
            str(spec.seed),
            "--is_training",
            "1",
            "--optimizer",
            hp["optimizer"],
            "--train_epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--batch_size",
            str(args.batch_size or hp["batch_size"]),
            "--learning_rate",
            str(hp["learning_rate"]),
            "--devices_ids",
            args.gpu or "0",
            "--root_path",
            str_path(root_path),
            "--data_path",
            ds["file"],
            "--data",
            data_name,
            "--period",
            str(ds["period"]),
            "--features",
            "M",
            "--c_in",
            str(ds["n_vars"]),
            "--history_len",
            str(seq_len),
            "--pred_len",
            str(spec.horizon),
            "--individual",
            str(hp["individual"]),
            "--norm_const",
            str(hp["norm_const"]),
            "--align_const",
            str(hp["align_const"]),
            "--vm_arch",
            hp["vm_arch"],
            "--ft_type",
            hp["ft_type"],
            "--trained_MAE_ckpt",
            str_path(ckpt),
        ]
        return cmd, cwd, log_path, {}

    if spec.model == "TimeLLM":
        hp = timellm_hparams(config, spec.dataset, args)
        cwd = REPO_ROOT / "baselines" / "Time-LLM"
        master_port = str(args.master_port or (29000 + (spec.seed % 1000)))
        cmd = [
            args.timellm_accelerate or os.environ.get("TIMELLM_ACCELERATE", "accelerate"),
            "launch",
            "--num_processes",
            str(hp["processes"]),
            "--mixed_precision",
            hp["mixed_precision"],
            "--main_process_port",
            master_port,
            "run_main.py",
            "--task_name",
            "long_term_forecast",
            "--is_training",
            "1",
            "--root_path",
            str_path(root_path),
            "--data_path",
            ds["file"],
            "--model_id",
            f"{spec.dataset}_{seq_len}_{spec.horizon}_seed{spec.seed}",
            "--model",
            "TimeLLM",
            "--data",
            ds.get("timellm_data", spec.dataset),
            "--features",
            "M",
            "--seq_len",
            str(seq_len),
            "--label_len",
            str(label_len),
            "--pred_len",
            str(spec.horizon),
            "--factor",
            "3",
            "--enc_in",
            str(ds["n_vars"]),
            "--dec_in",
            str(ds["n_vars"]),
            "--c_out",
            str(ds["n_vars"]),
            "--des",
            "Exp1",
            "--itr",
            "1",
            "--d_model",
            str(hp["d_model"]),
            "--d_ff",
            str(hp["d_ff"]),
            "--batch_size",
            str(hp["batch_size"]),
            "--learning_rate",
            str(hp["learning_rate"]),
            "--llm_model",
            hp["llm_model"],
            "--llm_dim",
            str(hp["llm_dim"]),
            "--llm_layers",
            str(hp["llm_layers"]),
            "--train_epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--seed",
            str(spec.seed),
            "--model_comment",
            f"Exp1-{spec.dataset}",
        ]
        return cmd, cwd, log_path, {}

    if spec.model == "T3Time":
        hp = t3_hparams(config, spec.dataset, spec.horizon, args)
        cwd = REPO_ROOT / "baselines" / "T3Time"
        extra_env = {
            "T3TIME_DATA_ROOT": str_path(REPO_ROOT / "data" / "t3time_dataset"),
            "T3TIME_EMBED_ROOT": str_path(REPO_ROOT / "data" / "t3time_embeddings"),
            "PYTHONPATH": str(cwd) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        }
        if args.t3_gpt2_model_path:
            extra_env["T3TIME_GPT2_MODEL_PATH"] = args.t3_gpt2_model_path
        if args.t3_gpt2_local_only:
            extra_env["T3TIME_GPT2_LOCAL_ONLY"] = "1"
        cmd = [
            python_for("T3Time", args),
            "-u",
            "train.py",
            "--data_path",
            ds["t3"],
            "--batch_size",
            str(hp["batch_size"]),
            "--num_nodes",
            str(ds["n_vars"]),
            "--seq_len",
            str(seq_len),
            "--pred_len",
            str(spec.horizon),
            "--epochs",
            str(epochs),
            "--es_patience",
            str(patience),
            "--seed",
            str(spec.seed),
            "--channel",
            str(hp["channel"]),
            "--learning_rate",
            str(hp["learning_rate"]),
            "--dropout_n",
            str(hp["dropout_n"]),
            "--e_layer",
            str(hp["e_layer"]),
            "--d_layer",
            str(hp["d_layer"]),
            "--num_workers",
            str(hp["num_workers"]),
            "--save",
            str_path(REPO_ROOT / "results" / "t3time_logs") + os.sep,
        ]
        return cmd, cwd, log_path, extra_env

    raise ValueError(f"Unknown model: {spec.model}")


def t3_embedding_commands(
    spec: RunSpec, config: dict, args: argparse.Namespace
) -> list[tuple[list[str], Path, Path, dict[str, str]]]:
    if spec.model != "T3Time" or args.skip_t3_embeddings:
        return []
    ds = config["datasets"][spec.dataset]
    mcfg = config["models"]["T3Time"]
    cwd = REPO_ROOT / "baselines" / "T3Time"
    embed_root = REPO_ROOT / "data" / "t3time_embeddings"
    extra_env = {
        "T3TIME_DATA_ROOT": str_path(REPO_ROOT / "data" / "t3time_dataset"),
        "T3TIME_EMBED_ROOT": str_path(embed_root),
        "PYTHONPATH": str(cwd) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    if args.t3_gpt2_model_path:
        extra_env["T3TIME_GPT2_MODEL_PATH"] = args.t3_gpt2_model_path
    if args.t3_gpt2_local_only:
        extra_env["T3TIME_GPT2_LOCAL_ONLY"] = "1"
    out: list[tuple[list[str], Path, Path, dict[str, str]]] = []
    for divide in ["train", "val", "test"]:
        marker = embed_root / ds["t3"] / f"seq{config['seq_len']}_pred{spec.horizon}" / divide / "0.h5"
        if marker.exists() and not args.force_t3_embeddings:
            continue
        log_path = (
            REPO_ROOT
            / "logs"
            / f"T3Time_embed_{spec.dataset}_h{spec.horizon}_{divide}_seed{spec.seed}.log"
        )
        cmd = [
            python_for("T3Time", args),
            "-u",
            "storage/store_emb.py",
            "--divide",
            divide,
            "--data_path",
            ds["t3"],
            "--num_nodes",
            str(ds["n_vars"]),
            "--input_len",
            str(config["seq_len"]),
            "--output_len",
            str(spec.horizon),
            "--device",
            "cuda" if args.gpu is None else "cuda:0",
            "--num_workers",
            str(args.t3_embed_num_workers),
            "--batch_size",
            str(args.t3_embed_batch_size or mcfg["default_embedding_batch_size"]),
        ]
        if args.t3_max_embed_samples:
            cmd += ["--max_samples", str(args.t3_max_embed_samples)]
        out.append((cmd, cwd, log_path, extra_env))
    return out


def parse_metrics(text: str, model: str) -> tuple[float | None, float | None]:
    if model in {"PatchTST", "DMMV"}:
        matches = re.findall(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)", text)
        if matches:
            mse, mae = matches[-1]
            return float(mse), float(mae)
    if model == "T3Time":
        matches = re.findall(
            r"On average horizons,\s*Test MSE:\s*([0-9.eE+-]+),\s*Test MAE:\s*([0-9.eE+-]+)",
            text,
        )
        if matches:
            mse, mae = matches[-1]
            return float(mse), float(mae)
    if model == "TimeLLM":
        matches = re.findall(
            r"Epoch:\s*\d+\s*\|\s*Train Loss:\s*[0-9.eE+-]+\s*"
            r"Vali Loss:\s*([0-9.eE+-]+)\s*Test Loss:\s*([0-9.eE+-]+)\s*"
            r"MAE Loss:\s*([0-9.eE+-]+)",
            text,
        )
        if matches:
            _, mse, mae = min(matches, key=lambda row: float(row[0]))
            return float(mse), float(mae)
    return None, None


def command_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "dataset",
        "horizon",
        "seed",
        "status",
        "mse",
        "mae",
        "seconds",
        "log_path",
        "command",
    ]
    exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> tuple[int, str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = f"[cwd] {cwd}\n[cmd] {command_str(cmd)}\n"
    if dry_run:
        print(printable)
        return 0, printable, 0.0

    t0 = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(printable)
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        chunks: list[str] = []
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            chunks.append(line)
        proc.wait()
    return proc.returncode, "".join(chunks), time.time() - t0


def iter_specs(config: dict, args: argparse.Namespace) -> list[RunSpec]:
    models = split_arg(args.models, MODEL_ORDER)
    datasets = split_arg(args.datasets, config["datasets"].keys())
    seeds = [int(x) for x in split_arg(args.seeds, [str(s) for s in config["seeds"]])]
    specs: list[RunSpec] = []
    for model in models:
        if model not in MODEL_ORDER:
            raise ValueError(f"Unknown model `{model}`. Choices: {','.join(MODEL_ORDER)}")
        for dataset in datasets:
            if dataset not in config["datasets"]:
                raise ValueError(f"Unknown dataset `{dataset}`")
            horizons = [int(x) for x in split_arg(args.horizons, config["datasets"][dataset]["horizons"])]
            for horizon in horizons:
                if horizon not in config["datasets"][dataset]["horizons"]:
                    continue
                for seed in seeds:
                    specs.append(RunSpec(model, dataset, horizon, seed))
    if args.max_runs:
        specs = specs[: args.max_runs]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four official baselines.")
    parser.add_argument("--config", default="configs/experiment1_fullshot.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset list.")
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds.")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value for single run.")
    parser.add_argument("--python", default=None, help="Default Python executable for non-accelerate runs.")
    parser.add_argument("--patchtst-python", default=None, help="Python executable for PatchTST.")
    parser.add_argument("--dmmv-python", default=None, help="Python executable for DMMV.")
    parser.add_argument("--t3time-python", default=None, help="Python executable for T3Time.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override training batch size.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument("--timellm-llm-model", default=None, choices=["LLAMA", "GPT2", "BERT"])
    parser.add_argument("--timellm-llm-layers", type=int, default=0)
    parser.add_argument("--timellm-processes", type=int, default=0)
    parser.add_argument("--timellm-mixed-precision", default=None)
    parser.add_argument("--timellm-accelerate", default=None, help="Path to the Time-LLM env's accelerate executable.")
    parser.add_argument("--skip-t3-embeddings", action="store_true")
    parser.add_argument("--force-t3-embeddings", action="store_true")
    parser.add_argument("--t3-embed-batch-size", type=int, default=0)
    parser.add_argument("--t3-embed-num-workers", type=int, default=4)
    parser.add_argument("--t3-max-embed-samples", type=int, default=0)
    parser.add_argument("--t3-gpt2-model-path", default=None)
    parser.add_argument("--t3-gpt2-local-only", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        raise SystemExit("Choose --dry-run or --execute.")

    mkdirs()
    config = load_config(REPO_ROOT / args.config)
    specs = iter_specs(config, args)
    print(f"Prepared {len(specs)} runs.")

    for idx, spec in enumerate(specs, 1):
        print(f"\n[{idx}/{len(specs)}] {spec.model} {spec.dataset} h={spec.horizon} seed={spec.seed}")
        for emb_cmd, emb_cwd, emb_log, emb_extra_env in t3_embedding_commands(spec, config, args):
            code, _, _ = run_command(
                emb_cmd,
                emb_cwd,
                emb_log,
                env_with_gpu(args.gpu, emb_extra_env),
                dry_run=args.dry_run,
            )
            if code != 0:
                print(f"[failed] T3Time embedding command exited with {code}: {emb_log}")
                if args.fail_fast:
                    raise SystemExit(code)

        cmd, cwd, log_path, extra_env = command_for(spec, config, args)
        code, text, seconds = run_command(
            cmd,
            cwd,
            log_path,
            env_with_gpu(args.gpu, extra_env),
            dry_run=args.dry_run,
        )
        mse, mae = parse_metrics(text, spec.model)
        status = "dry-run" if args.dry_run else ("ok" if code == 0 else f"failed:{code}")
        if args.execute:
            append_result(
                {
                    "model": spec.model,
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "seed": spec.seed,
                    "status": status,
                    "mse": "" if mse is None else mse,
                    "mae": "" if mae is None else mae,
                    "seconds": f"{seconds:.2f}",
                    "log_path": str(log_path.relative_to(REPO_ROOT)),
                    "command": command_str(cmd),
                }
            )
        print(f"[result] status={status} mse={mse} mae={mae} log={log_path}")
        if code != 0 and args.fail_fast:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
