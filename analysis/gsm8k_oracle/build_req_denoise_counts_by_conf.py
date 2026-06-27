#!/usr/bin/env python3

"""Build per-request denoise-row counts across confidence runs.

This script reads step_data.json files from the 20260422 experiment runs and
writes a JSON mapping of:

    { req_id: {"0.9": count, "0.8": count, ...} }

Warmup request ids are ignored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CONF_DIRS: dict[str, str] = {
    "0.9": "0_conf_0_9",
    "0.8": "1_conf_0_8",
    "0.7": "2_conf_0_7",
    "0.6": "3_conf_0_6",
    "0.5": "4_conf_0_5",
}

BASE_DIR = Path("/u/tchang85/dllm/sbatch_experiment_dir/20260422/210041_get_ground_truth")
DEFAULT_OUTPUT = Path("/u/tchang85/dllm/analysis/gsm8k_oracle/req_denoise_counts_by_conf.json")


def iter_step_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(dict)

    for conf, dirname in CONF_DIRS.items():
        step_path = BASE_DIR / dirname / "step_data.json"
        if not step_path.exists():
            raise FileNotFoundError(f"Missing step_data.json: {step_path}")

        per_request: dict[str, int] = defaultdict(int)
        for row in iter_step_rows(step_path):
            req_id = row.get("id")
            if not req_id or str(req_id).startswith("cmpl-warmup"):
                continue
            per_request[str(req_id)] += 1

        for req_id, count in per_request.items():
            counts[req_id][conf] = count

    return dict(sorted(counts.items()))


def main() -> None:
    counts = build_counts()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
