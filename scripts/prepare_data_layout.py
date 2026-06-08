#!/usr/bin/env python3
"""Prepare the shared dataset layout used by the four official baselines."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def same_path(src: Path, dst: Path) -> bool:
    try:
        return src.resolve() == dst.resolve()
    except FileNotFoundError:
        return False


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> str:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if same_path(src, dst):
            return "exists"
        if not overwrite:
            return "exists"
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode in {"symlink", "auto"}:
        try:
            rel = os.path.relpath(src, dst.parent)
            dst.symlink_to(rel)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise

    shutil.copy2(src, dst)
    return "copy"


def candidate_source(source_root: Path | None, rel_root: str, file_name: str) -> Path:
    if source_root is None:
        return REPO_ROOT / rel_root / file_name
    return source_root / Path(rel_root).relative_to("data") / file_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared dataset paths.")
    parser.add_argument("--config", default="configs/experiment1_fullshot.yaml")
    parser.add_argument(
        "--source-root",
        default=None,
        help="Directory containing ETT-small/, weather/, electricity/, traffic/, illness/, exchange_rate/.",
    )
    parser.add_argument("--mode", choices=["auto", "symlink", "copy"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None

    missing: list[str] = []
    for dataset, spec in config["datasets"].items():
        dst = REPO_ROOT / spec["root"] / spec["file"]
        src = candidate_source(source_root, spec["root"], spec["file"])
        if not src.exists():
            missing.append(f"{dataset}: {src}")
            continue
        if args.check_only:
            print(f"[ok] {dataset}: {src}")
            continue
        action = link_or_copy(src, dst, args.mode, args.overwrite)
        print(f"[{action}] {dataset}: {dst}")

    if missing:
        print("\nMissing dataset files:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)

    if args.check_only:
        return

    t3_root = REPO_ROOT / "data" / "t3time_dataset"
    for dataset, spec in config["datasets"].items():
        src = REPO_ROOT / spec["root"] / spec["file"]
        dst = t3_root / f"{spec['t3']}.csv"
        action = link_or_copy(src, dst, args.mode, args.overwrite)
        print(f"[{action}] T3Time {dataset}: {dst}")


if __name__ == "__main__":
    main()
