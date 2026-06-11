#!/usr/bin/env python3
"""Summarize GPU memory probe records."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize memory probe results.")
    parser.add_argument("--input", default="results/memory_probe.csv")
    parser.add_argument("--output-dir", default="results/summary")
    args = parser.parse_args()

    src = REPO_ROOT / args.input
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if df.empty:
        raise SystemExit("No memory probe rows found.")

    for col in ["peak_mem_mb", "total_mem_mb", "peak_mem_ratio", "oom", "memory_anomaly"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    case_summary = (
        df.groupby(["model", "dataset", "horizon", "seed"], as_index=False)
        .agg(
            any_failed=("status", lambda s: int(any(str(x).startswith("failed") for x in s))),
            any_oom=("oom", "max"),
            any_memory_anomaly=("memory_anomaly", "max"),
            peak_mem_mb=("peak_mem_mb", "max"),
            total_mem_mb=("total_mem_mb", "max"),
            peak_mem_ratio=("peak_mem_ratio", "max"),
            phases=("phase", lambda s: ";".join(map(str, s))),
        )
        .sort_values(["any_memory_anomaly", "peak_mem_ratio"], ascending=[False, False])
    )

    model_summary = (
        case_summary.groupby("model", as_index=False)
        .agg(
            probed_cases=("dataset", "count"),
            failed_cases=("any_failed", "sum"),
            oom_cases=("any_oom", "sum"),
            memory_anomaly_cases=("any_memory_anomaly", "sum"),
            max_peak_mem_mb=("peak_mem_mb", "max"),
            max_peak_mem_ratio=("peak_mem_ratio", "max"),
        )
        .sort_values(["memory_anomaly_cases", "max_peak_mem_ratio"], ascending=[False, False])
    )

    case_summary.to_csv(out_dir / "memory_probe_case_summary.csv", index=False)
    model_summary.to_csv(out_dir / "memory_probe_model_summary.csv", index=False)
    print(f"[saved] {out_dir / 'memory_probe_case_summary.csv'}")
    print(f"[saved] {out_dir / 'memory_probe_model_summary.csv'}")


if __name__ == "__main__":
    main()
