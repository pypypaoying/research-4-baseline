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
MEMORY_PROBE_CSV = REPO_ROOT / "results" / "memory_probe.csv"
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


def t3_use_probe_embedding_cache(args: argparse.Namespace) -> bool:
    return bool(args.memory_probe or (args.t3_embedding_only and args.t3_max_embed_samples))


def t3_embedding_root(args: argparse.Namespace) -> Path:
    if t3_use_probe_embedding_cache(args):
        return REPO_ROOT / "data" / "t3time_embeddings_probe"
    return REPO_ROOT / "data" / "t3time_embeddings"


def t3_embedding_split_dir(embed_root: Path, dataset: str, seq_len: int, divide: str) -> Path:
    return embed_root / dataset / f"seq{seq_len}" / divide


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def t3_embedding_cache_ready(split_dir: Path, max_samples: int, force: bool) -> bool:
    if force:
        return False
    if not max_samples:
        return False
    meta = read_json(split_dir / "_meta.json")
    if not meta:
        return False
    written = int(meta.get("cached_prefix_samples", meta.get("written_samples", 0)) or 0)
    return written >= max_samples


def mkdirs() -> None:
    for rel in [
        "logs",
        "results",
        "checkpoints",
        "data/t3time_dataset",
        "data/t3time_embeddings",
        "data/t3time_embeddings_probe",
        "results/gpu_traces",
    ]:
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


def effective_epochs(config: dict, args: argparse.Namespace) -> int:
    if args.memory_probe:
        return args.probe_epochs
    return int(config["epochs"])


def effective_patience(config: dict, args: argparse.Namespace) -> int:
    if args.memory_probe:
        return max(1, min(args.probe_epochs, int(config["patience"])))
    return int(config["patience"])


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
    elif overrides.memory_probe:
        needed = max(1, overrides.probe_max_train_batches, overrides.probe_max_eval_batches)
        hp["batch_size"] = min(hp["batch_size"], max(1, overrides.t3_max_embed_samples // needed))
    return hp


def effective_t3_embed_max_samples(spec: RunSpec, config: dict, args: argparse.Namespace) -> int:
    if not args.t3_max_embed_samples:
        return 0
    min_samples = max(1, args.probe_max_train_batches, args.probe_max_eval_batches)
    if args.memory_probe:
        hp = t3_hparams(config, spec.dataset, spec.horizon, args)
        min_samples *= int(hp["batch_size"])
    return max(args.t3_max_embed_samples, min_samples)


def command_for(spec: RunSpec, config: dict, args: argparse.Namespace) -> tuple[list[str], Path, Path, dict[str, str]]:
    ds = config["datasets"][spec.dataset]
    seq_len = int(config["seq_len"])
    label_len = int(config["label_len"])
    epochs = effective_epochs(config, args)
    patience = effective_patience(config, args)
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
        extra_env = {}
        if args.memory_probe:
            extra_env = {
                "DMMV_MAX_TRAIN_BATCHES": str(args.probe_max_train_batches),
                "DMMV_MAX_EVAL_BATCHES": str(args.probe_max_eval_batches),
                "DMMV_MAX_TEST_BATCHES": str(args.probe_max_eval_batches),
            }
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
        return cmd, cwd, log_path, extra_env

    if spec.model == "TimeLLM":
        hp = timellm_hparams(config, spec.dataset, args)
        cwd = REPO_ROOT / "baselines" / "Time-LLM"
        master_port = str(args.master_port or (29000 + (spec.seed % 1000)))
        extra_env = {}
        if int(hp["processes"]) == 1:
            extra_env.update(
                {
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": master_port,
                }
            )
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
        return cmd, cwd, log_path, extra_env

    if spec.model == "T3Time":
        hp = t3_hparams(config, spec.dataset, spec.horizon, args)
        cwd = REPO_ROOT / "baselines" / "T3Time"
        embed_root = t3_embedding_root(args)
        extra_env = {
            "T3TIME_DATA_ROOT": str_path(REPO_ROOT / "data" / "t3time_dataset"),
            "T3TIME_EMBED_ROOT": str_path(embed_root),
            "T3TIME_PROMPT_BATCH_SIZE": str(args.t3_prompt_batch_size),
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
        if args.memory_probe:
            cmd += [
                "--max_train_batches",
                str(args.probe_max_train_batches),
                "--max_eval_batches",
                str(args.probe_max_eval_batches),
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
    embed_root = t3_embedding_root(args)
    extra_env = {
        "T3TIME_DATA_ROOT": str_path(REPO_ROOT / "data" / "t3time_dataset"),
        "T3TIME_EMBED_ROOT": str_path(embed_root),
        "T3TIME_PROMPT_BATCH_SIZE": str(args.t3_prompt_batch_size),
        "PYTHONPATH": str(cwd) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    if args.t3_gpt2_model_path:
        extra_env["T3TIME_GPT2_MODEL_PATH"] = args.t3_gpt2_model_path
    if args.t3_gpt2_local_only:
        extra_env["T3TIME_GPT2_LOCAL_ONLY"] = "1"
    out: list[tuple[list[str], Path, Path, dict[str, str]]] = []
    max_samples = effective_t3_embed_max_samples(spec, config, args)
    for divide in ["train", "val", "test"]:
        split_dir = t3_embedding_split_dir(
            embed_root,
            ds["t3"],
            int(config["seq_len"]),
            divide,
        )
        if t3_embedding_cache_ready(split_dir, max_samples, args.force_t3_embeddings):
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
            "--prompt_batch_size",
            str(args.t3_prompt_batch_size),
        ]
        if max_samples:
            cmd += ["--max_samples", str(max_samples)]
        if args.force_t3_embeddings:
            cmd += ["--force"]
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


def append_memory_probe(row: dict) -> None:
    MEMORY_PROBE_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase",
        "model",
        "dataset",
        "horizon",
        "seed",
        "status",
        "oom",
        "memory_anomaly",
        "peak_mem_mb",
        "total_mem_mb",
        "peak_mem_ratio",
        "seconds",
        "trace_path",
        "log_path",
        "command",
    ]
    exists = MEMORY_PROBE_CSV.exists()
    with MEMORY_PROBE_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def nvidia_smi_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def start_gpu_trace(trace_path: Path, interval_ms: int) -> subprocess.Popen | None:
    if not nvidia_smi_available():
        return None
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu",
        "--format=csv",
        "-lms",
        str(interval_ms),
    ]
    f = trace_path.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL, text=True)
    proc._trace_file = f  # type: ignore[attr-defined]
    return proc


def stop_gpu_trace(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    trace_file = getattr(proc, "_trace_file", None)
    if trace_file is not None:
        trace_file.close()


def parse_mem_mb(value: str) -> int | None:
    match = re.search(r"([0-9]+)", value)
    return int(match.group(1)) if match else None


def summarize_gpu_trace(trace_path: Path) -> tuple[int | None, int | None, float | None]:
    if not trace_path.exists():
        return None, None, None
    peak = None
    total_at_peak = None
    with trace_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.lower().startswith("timestamp") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            used = parse_mem_mb(parts[2])
            total = parse_mem_mb(parts[3])
            if used is None or total is None:
                continue
            if peak is None or used > peak:
                peak = used
                total_at_peak = total
    ratio = None
    if peak is not None and total_at_peak:
        ratio = peak / total_at_peak
    return peak, total_at_peak, ratio


def detect_oom(text: str, returncode: int) -> bool:
    patterns = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "oom",
        "memoryerror",
    ]
    lower = text.lower()
    return any(p in lower for p in patterns)


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
    gpu_trace_path: Path | None = None,
    gpu_sample_interval_ms: int = 500,
) -> tuple[int, str, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = f"[cwd] {cwd}\n[cmd] {command_str(cmd)}\n"
    probe_env = {
        key: env[key]
        for key in sorted(env)
        if key.startswith("DMMV_MAX_")
        or key.startswith("T3TIME_")
        or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"}
    }
    if probe_env:
        printable += f"[env] {probe_env}\n"
    if dry_run:
        print(printable)
        return 0, printable, 0.0

    t0 = time.time()
    trace_proc = start_gpu_trace(gpu_trace_path, gpu_sample_interval_ms) if gpu_trace_path else None
    proc = None
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        try:
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
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
                chunks.append(line)
            proc.wait()
        except OSError as exc:
            msg = f"[runner-error] failed to start command: {exc}\n"
            print(msg, end="")
            f.write(msg)
            chunks.append(msg)
            return 127, "".join(chunks), time.time() - t0
        finally:
            stop_gpu_trace(trace_proc)
    return (proc.returncode if proc is not None else 127), "".join(chunks), time.time() - t0


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


def record_memory_probe(
    phase: str,
    spec: RunSpec,
    status: str,
    text: str,
    returncode: int,
    seconds: float,
    trace_path: Path,
    log_path: Path,
    command: list[str],
    anomaly_threshold: float,
) -> bool:
    peak_mb, total_mb, ratio = summarize_gpu_trace(trace_path)
    oom = detect_oom(text, returncode)
    anomaly = bool(oom or (ratio is not None and ratio >= anomaly_threshold))
    append_memory_probe(
        {
            "phase": phase,
            "model": spec.model,
            "dataset": spec.dataset,
            "horizon": spec.horizon,
            "seed": spec.seed,
            "status": status,
            "oom": int(oom),
            "memory_anomaly": int(anomaly),
            "peak_mem_mb": "" if peak_mb is None else peak_mb,
            "total_mem_mb": "" if total_mb is None else total_mb,
            "peak_mem_ratio": "" if ratio is None else f"{ratio:.4f}",
            "seconds": f"{seconds:.2f}",
            "trace_path": str(trace_path.relative_to(REPO_ROOT)),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
            "command": command_str(command),
        }
    )
    print(
        "[memory] "
        f"phase={phase} peak={peak_mb}/{total_mb}MB "
        f"ratio={None if ratio is None else round(ratio, 4)} oom={oom} anomaly={anomaly}"
    )
    return anomaly


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
    parser.add_argument(
        "--memory-probe",
        action="store_true",
        help="Short-run commands and record GPU memory traces instead of full result metrics.",
    )
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
    parser.add_argument(
        "--t3-embedding-only",
        action="store_true",
        help="Only generate T3Time prompt embeddings, useful for diagnosing the GPT-2 preprocessing stage.",
    )
    parser.add_argument("--t3-embed-batch-size", type=int, default=0)
    parser.add_argument("--t3-embed-num-workers", type=int, default=4)
    parser.add_argument(
        "--t3-prompt-batch-size",
        type=int,
        default=32,
        help="Number of T3Time GPT-2 prompts embedded per forward pass.",
    )
    parser.add_argument("--t3-max-embed-samples", type=int, default=0)
    parser.add_argument("--t3-gpt2-model-path", default=None)
    parser.add_argument("--t3-gpt2-local-only", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=1)
    parser.add_argument("--probe-max-train-batches", type=int, default=2)
    parser.add_argument("--probe-max-eval-batches", type=int, default=1)
    parser.add_argument("--probe-t3-max-embed-samples", type=int, default=8)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=500)
    parser.add_argument("--memory-anomaly-threshold", type=float, default=0.90)
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        raise SystemExit("Choose --dry-run or --execute.")
    if args.t3_embedding_only:
        models = split_arg(args.models, ["T3Time"])
        if models != ["T3Time"]:
            raise SystemExit("--t3-embedding-only is only valid with --models T3Time.")
        args.models = "T3Time"
    if args.memory_probe and args.t3_max_embed_samples == 0:
        args.t3_max_embed_samples = args.probe_t3_max_embed_samples

    mkdirs()
    config = load_config(REPO_ROOT / args.config)
    specs = iter_specs(config, args)
    print(f"Prepared {len(specs)} runs.")

    for idx, spec in enumerate(specs, 1):
        print(f"\n[{idx}/{len(specs)}] {spec.model} {spec.dataset} h={spec.horizon} seed={spec.seed}")
        for emb_idx, (emb_cmd, emb_cwd, emb_log, emb_extra_env) in enumerate(
            t3_embedding_commands(spec, config, args), 1
        ):
            emb_trace = (
                REPO_ROOT
                / "results"
                / "gpu_traces"
                / f"{spec.model}_{spec.dataset}_h{spec.horizon}_seed{spec.seed}_embedding{emb_idx}.csv"
            )
            code, emb_text, emb_seconds = run_command(
                emb_cmd,
                emb_cwd,
                emb_log,
                env_with_gpu(args.gpu, emb_extra_env),
                dry_run=args.dry_run,
                gpu_trace_path=emb_trace if args.memory_probe and args.execute else None,
                gpu_sample_interval_ms=args.gpu_sample_interval_ms,
            )
            if args.memory_probe and args.execute:
                record_memory_probe(
                    phase=f"t3_embedding_{emb_idx}",
                    spec=spec,
                    status="ok" if code == 0 else f"failed:{code}",
                    text=emb_text,
                    returncode=code,
                    seconds=emb_seconds,
                    trace_path=emb_trace,
                    log_path=emb_log,
                    command=emb_cmd,
                    anomaly_threshold=args.memory_anomaly_threshold,
                )
            if code != 0:
                print(f"[failed] T3Time embedding command exited with {code}: {emb_log}")
                if args.fail_fast:
                    raise SystemExit(code)

        if args.t3_embedding_only:
            print("[embedding-only] Skipping training command.")
            continue

        cmd, cwd, log_path, extra_env = command_for(spec, config, args)
        trace_path = (
            REPO_ROOT
            / "results"
            / "gpu_traces"
            / f"{spec.model}_{spec.dataset}_h{spec.horizon}_seed{spec.seed}_train.csv"
        )
        code, text, seconds = run_command(
            cmd,
            cwd,
            log_path,
            env_with_gpu(args.gpu, extra_env),
            dry_run=args.dry_run,
            gpu_trace_path=trace_path if args.memory_probe and args.execute else None,
            gpu_sample_interval_ms=args.gpu_sample_interval_ms,
        )
        mse, mae = parse_metrics(text, spec.model)
        status = "dry-run" if args.dry_run else ("ok" if code == 0 else f"failed:{code}")
        if args.memory_probe and args.execute:
            record_memory_probe(
                phase="train",
                spec=spec,
                status=status,
                text=text,
                returncode=code,
                seconds=seconds,
                trace_path=trace_path,
                log_path=log_path,
                command=cmd,
                anomaly_threshold=args.memory_anomaly_threshold,
            )
        elif args.execute:
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
