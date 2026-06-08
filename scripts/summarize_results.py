#!/usr/bin/env python3
"""Summarize unified full-shot results from results/runs.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def mean_numeric(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        df.groupby(keys, as_index=False)
        .agg(mse=("mse", "mean"), mae=("mae", "mean"), n_seeds=("seed", "nunique"))
        .sort_values(keys)
    )


def add_ranks(per_case: pd.DataFrame) -> pd.DataFrame:
    ranked = per_case.copy()
    ranked["mse_rank"] = ranked.groupby(["dataset", "horizon"])["mse"].rank(method="min")
    ranked["mae_rank"] = ranked.groupby(["dataset", "horizon"])["mae"].rank(method="min")
    ranked["is_first_mse"] = ranked["mse_rank"].eq(1)
    ranked["is_first_mae"] = ranked["mae_rank"].eq(1)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize four-baseline experiment results.")
    parser.add_argument("--input", default="results/runs.csv")
    parser.add_argument("--output-dir", default="results/summary")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    input_path = REPO_ROOT / args.input
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df[df["status"].eq("ok")].copy()
    df["mse"] = pd.to_numeric(df["mse"], errors="coerce")
    df["mae"] = pd.to_numeric(df["mae"], errors="coerce")
    df = df.dropna(subset=["mse", "mae"])

    if df.empty:
        raise SystemExit("No successful metric rows found.")

    per_case = mean_numeric(df, ["model", "dataset", "horizon"])
    if args.require_complete and (per_case["n_seeds"] < 3).any():
        missing = per_case[per_case["n_seeds"] < 3]
        raise SystemExit(f"Incomplete seed coverage:\n{missing}")

    ranked = add_ranks(per_case)
    dataset_avg = mean_numeric(per_case, ["model", "dataset"])
    horizon_avg = mean_numeric(per_case, ["model", "horizon"])
    overall = (
        ranked.groupby("model", as_index=False)
        .agg(
            avg_mse=("mse", "mean"),
            avg_mae=("mae", "mean"),
            overall_rank_mse=("mse_rank", "mean"),
            overall_rank_mae=("mae_rank", "mean"),
            first_count_mse=("is_first_mse", "sum"),
            first_count_mae=("is_first_mae", "sum"),
            n_cases=("mse", "count"),
        )
        .sort_values(["overall_rank_mse", "avg_mse", "avg_mae"])
    )

    per_case.to_csv(out_dir / "per_dataset_horizon_seedavg.csv", index=False)
    ranked.to_csv(out_dir / "per_dataset_horizon_ranked.csv", index=False)
    dataset_avg.to_csv(out_dir / "dataset_average.csv", index=False)
    horizon_avg.to_csv(out_dir / "horizon_average.csv", index=False)
    overall.to_csv(out_dir / "overall_rank_first_count.csv", index=False)

    print(f"[saved] {out_dir / 'per_dataset_horizon_seedavg.csv'}")
    print(f"[saved] {out_dir / 'dataset_average.csv'}")
    print(f"[saved] {out_dir / 'horizon_average.csv'}")
    print(f"[saved] {out_dir / 'overall_rank_first_count.csv'}")


if __name__ == "__main__":
    main()
