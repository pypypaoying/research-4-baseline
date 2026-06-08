#!/usr/bin/env python3
"""Download lightweight checkpoints required by official baseline scripts."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DMMV_MAE_URL = "https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth"


def download(url: str, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        print(f"[exists] {dst}")
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"[saved] {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download baseline checkpoints.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dmmv-mae-path",
        default="checkpoints/mae_visualize_vit_base.pth",
        help="Path relative to repository root.",
    )
    args = parser.parse_args()
    download(DMMV_MAE_URL, REPO_ROOT / args.dmmv_mae_path, args.overwrite)


if __name__ == "__main__":
    main()
