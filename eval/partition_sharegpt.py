#!/usr/bin/env python3
"""Randomly partition ShareGPT JSONL into train and test splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def partition_sharegpt(
    input_path: Path, train_size: int, seed: int
) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(input_path)
    total = len(rows)

    if train_size > total:
        raise ValueError(
            f"train_size ({train_size}) cannot exceed total samples ({total})"
        )

    rng = random.Random(seed)
    train_indices = set(rng.sample(range(total), train_size))

    train_rows = [row for i, row in enumerate(rows) if i in train_indices]
    test_rows = [row for i, row in enumerate(rows) if i not in train_indices]
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly partition sharegpt_gpt4_clean.jsonl into train/test splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sharegpt_gpt4_clean.jsonl"),
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=500,
        help="Number of samples to save into the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("sharegpt_gpt4_clean_train.jsonl"),
        help="Output path for the train split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("sharegpt_gpt4_clean_test.jsonl"),
        help="Output path for the test split.",
    )
    args = parser.parse_args()

    train_rows, test_rows = partition_sharegpt(args.input, args.train_size, args.seed)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)

    print(f"Loaded {len(train_rows) + len(test_rows)} samples from {args.input}")
    print(f"Saved {len(train_rows)} samples to {args.train_output}")
    print(f"Saved {len(test_rows)} samples to {args.test_output}")


if __name__ == "__main__":
    main()