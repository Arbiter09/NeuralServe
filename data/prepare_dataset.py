"""Download tatsu-lab/alpaca, shuffle, slice to N samples, and save as JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def prepare(output: str = "data/alpaca_12k.jsonl", size: int = 12000, seed: int = 42) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    print(f"Downloading tatsu-lab/alpaca …")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    records = list(ds)
    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:size]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for rec in records:
            row = {
                "instruction": rec.get("instruction", ""),
                "input": rec.get("input", ""),
                "output": rec.get("output", ""),
            }
            f.write(json.dumps(row) + "\n")

    # Statistics
    has_input = sum(1 for r in records if r.get("input", "").strip())
    avg_out_len = sum(len(r.get("output", "")) for r in records) / len(records)
    avg_inst_len = sum(len(r.get("instruction", "")) for r in records) / len(records)

    print(f"\n{'─'*50}")
    print(f"  Saved {len(records):,} samples → {out_path}")
    print(f"  Examples with non-empty input : {has_input:,} ({has_input/len(records):.1%})")
    print(f"  Avg instruction length        : {avg_inst_len:.0f} chars")
    print(f"  Avg output length             : {avg_out_len:.0f} chars")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset for NeuralServe training")
    parser.add_argument("--output", default="data/alpaca_12k.jsonl", help="Output JSONL path")
    parser.add_argument("--size", type=int, default=12000, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    prepare(output=args.output, size=args.size, seed=args.seed)
