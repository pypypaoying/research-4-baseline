#!/usr/bin/env python3
"""Summarize T3Time prompt-embedding speed benchmark logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_int_flag(text: str, flag: str) -> int | None:
    match = re.search(rf"{re.escape(flag)}\s+([0-9]+)", text)
    return int(match.group(1)) if match else None


def parse_log(path: Path, n_vars: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    generated = 0
    ready = None
    required = None
    status = "unknown"

    generated_match = re.search(
        r"Generated\s+([0-9]+)\s+embedding samples.*?cache ready\s+([0-9]+)/([0-9]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if generated_match:
        generated = int(generated_match.group(1))
        ready = int(generated_match.group(2))
        required = int(generated_match.group(3))
        status = "generated"

    cache_match = re.search(
        r"Embedding cache ready.*?:\s+([0-9]+)/([0-9]+)\s+samples",
        text,
        flags=re.IGNORECASE,
    )
    if cache_match:
        generated = 0
        ready = int(cache_match.group(1))
        required = int(cache_match.group(2))
        status = "cache_hit"

    time_match = re.search(r"Total time spent:\s*([0-9.]+)\s*minutes", text)
    minutes = float(time_match.group(1)) if time_match else None
    seconds = minutes * 60.0 if minutes is not None else None

    split = "unknown"
    split_match = re.search(r"_(train|val|test)\.log$", path.name)
    if split_match:
        split = split_match.group(1)

    case = path.parent.name
    prompt_batch = parse_int_flag(text, "--prompt_batch_size")
    embed_batch = parse_int_flag(text, "--batch_size")
    max_samples = parse_int_flag(text, "--max_samples")
    generated_prompts = generated * n_vars
    samples_per_min = ""
    prompts_per_sec = ""
    if seconds and seconds > 0:
        samples_per_min = f"{generated / seconds * 60.0:.4f}"
        prompts_per_sec = f"{generated_prompts / seconds:.4f}"

    return {
        "case": case,
        "split": split,
        "status": status,
        "generated_samples": generated,
        "ready_samples": "" if ready is None else ready,
        "required_samples": "" if required is None else required,
        "generated_prompts_est": generated_prompts,
        "seconds": "" if seconds is None else f"{seconds:.2f}",
        "minutes": "" if minutes is None else f"{minutes:.4f}",
        "samples_per_min": samples_per_min,
        "prompts_per_sec_est": prompts_per_sec,
        "embedding_batch_size": "" if embed_batch is None else embed_batch,
        "prompt_batch_size": "" if prompt_batch is None else prompt_batch,
        "max_samples": "" if max_samples is None else max_samples,
        "log_path": str(path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize T3Time embedding benchmark logs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n-vars", type=int, default=862)
    args = parser.parse_args()

    input_dir = (REPO_ROOT / args.input_dir).resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    rows = []
    for log_path in sorted(input_dir.glob("*/*.log")):
        rows.append(parse_log(log_path, args.n_vars))

    if not rows:
        raise SystemExit(f"No benchmark logs found under {input_dir}")

    output = Path(args.output) if args.output else input_dir / "summary.csv"
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case",
        "split",
        "status",
        "generated_samples",
        "ready_samples",
        "required_samples",
        "generated_prompts_est",
        "seconds",
        "minutes",
        "samples_per_min",
        "prompts_per_sec_est",
        "embedding_batch_size",
        "prompt_batch_size",
        "max_samples",
        "log_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {output}")
    print("case,split,status,minutes,samples_per_min,prompts_per_sec_est")
    for row in rows:
        print(
            ",".join(
                str(row[key])
                for key in [
                    "case",
                    "split",
                    "status",
                    "minutes",
                    "samples_per_min",
                    "prompts_per_sec_est",
                ]
            )
        )


if __name__ == "__main__":
    main()
