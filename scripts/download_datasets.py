#!/usr/bin/env python3
"""Download the nine LTSF datasets from the THUML Time-Series-Library dataset."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from exc

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "thuml/Time-Series-Library"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark CSV files.")
    parser.add_argument("--config", default="configs/experiment1_fullshot.yaml")
    parser.add_argument("--repo-id", default=HF_REPO)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prepare-t3time",
        action="store_true",
        help="Run prepare_data_layout.py after downloading to create T3Time aliases.",
    )
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    for dataset, spec in config["datasets"].items():
        filename = f"{Path(spec['root']).relative_to('data').as_posix()}/{spec['file']}"
        target = REPO_ROOT / spec["root"] / spec["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.overwrite:
            print(f"[exists] {dataset}: {target}")
            continue
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=args.cache_dir,
        )
        shutil.copy2(downloaded, target)
        print(f"[downloaded] {dataset}: {target}")

    if args.prepare_t3time:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "prepare_data_layout.py"), "--overwrite"],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
