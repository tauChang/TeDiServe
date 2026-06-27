#!/usr/bin/env python3

"""Compare LM Eval sample outputs across multiple result files.

For each document (matched by doc_hash when available, otherwise doc_id),
assert that the responses are identical across all files. If not, print the
first mismatch for that document and exit with a non-zero status.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


FILTER_NAME = "flexible-extract"


def load_samples(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples_by_task = data.get("samples", {})
    if not isinstance(samples_by_task, dict):
        raise ValueError(f"{path}: expected top-level 'samples' to be a dict")

    samples: dict[str, dict[str, Any]] = {}
    for task_name, task_samples in samples_by_task.items():
        if not isinstance(task_samples, list):
            raise ValueError(f"{path}: expected samples[{task_name!r}] to be a list")
        for sample in task_samples:
            if not isinstance(sample, dict):
                raise ValueError(f"{path}: expected each sample to be a dict")
            if sample.get("filter") != FILTER_NAME:
                continue
            doc_hash = sample.get("doc_hash")
            doc_id = sample.get("doc_id")
            if doc_hash is not None:
                key = f"{task_name}:doc_hash:{doc_hash}"
            elif doc_id is not None:
                key = f"{task_name}:doc_id:{doc_id}"
            else:
                raise ValueError(
                    f"{path}: sample is missing both doc_hash and doc_id: {sample!r}"
                )
            samples[key] = sample

    return samples


def normalize_response(sample: dict[str, Any]) -> Any:
    if "resps" in sample:
        return sample["resps"]
    if "filtered_resps" in sample:
        return sample["filtered_resps"]
    raise ValueError(f"sample is missing both 'resps' and 'filtered_resps': {sample!r}")


def get_exact_match(sample: dict[str, Any]) -> Any:
    return sample.get("exact_match")


def main() -> int:
    # parser = argparse.ArgumentParser(
    #     description="Assert that identical docs have identical responses across LM Eval result files."
    # )
    # parser.add_argument("files", nargs="+", help="Result JSON files to compare")
    # args = parser.parse_args()

    # paths = [Path(file_path) for file_path in args.files]
    paths = [
        "/u/tchang85/dllm/experiment_dir/20260416/165239/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32.json",
        "/u/tchang85/dllm/experiment_dir/20260416/164947/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32.json",
        "/u/tchang85/dllm/experiment_dir/20260416/164652/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32.json",

        # cache, no batching
        # "/u/tchang85/dllm/experiment_dir/20260416/192609/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32.json",
        # "/u/tchang85/dllm/experiment_dir/20260416/192946/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block32.json",

        # no cache, no batching
        # "/u/tchang85/dllm/experiment_dir/20260416/193327/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
        # "/u/tchang85/dllm/experiment_dir/20260416/193842/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",

        # cache, batching
        # "/u/tchang85/dllm/experiment_dir/20260416/194416/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
        # "/u/tchang85/dllm/experiment_dir/20260416/194740/results/gsm8k_1/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",

    ]
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise SystemExit("Need at least two files to compare.")

    per_file_samples = {path: load_samples(path) for path in paths}

    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path, samples in per_file_samples.items():
        for key, sample in samples.items():
            grouped[key].append((path, sample))

    mismatches = 0
    for doc_key, entries in sorted(grouped.items()):
        if len(entries) != len(paths):
            missing = sorted(str(path) for path in set(paths) - {path for path, _ in entries})
            print(f"MISSING {doc_key}: missing in {missing}")
            mismatches += 1
            continue

        baseline_path, baseline_sample = entries[0]
        baseline_resp = normalize_response(baseline_sample)
        baseline_exact_match = get_exact_match(baseline_sample)

        for path, sample in entries[1:]:
            resp = normalize_response(sample)
            if resp != baseline_resp:
                print(f"MISMATCH {doc_key}")
                print(
                    f"  {baseline_path}: exact_match={baseline_exact_match} resp={json.dumps(baseline_resp, ensure_ascii=False)}"
                )
                print(
                    f"  {path}: exact_match={get_exact_match(sample)} resp={json.dumps(resp, ensure_ascii=False)}"
                )
                print("\n")
                mismatches += 1
                break

    if mismatches:
        print(f"Found {mismatches} mismatching or missing doc(s) out of {len(grouped)} total.")
        return 1

    print(f"All {len(grouped)} docs matched across {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())