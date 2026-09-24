#!/usr/bin/env python3
"""Small preflight check for the documented TeDiServe artifact modes."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tediserve-matplotlib")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "gpu"), required=True)
    args = parser.parse_args()

    require((3, 9) <= sys.version_info[:2] <= (3, 12),
            "Python 3.9–3.12 is supported by this source tree.")

    missing = []
    for name in ("lightgbm", "ray", "transformers", "vllm"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # report all common setup failures together
            missing.append(f"{name} ({exc})")
    require(not missing, "Missing or broken imports: " + "; ".join(missing))

    require((REPO_ROOT / "step_estimator_models/lgb_gsm8k/model.bin").is_file(),
            "The checked-in step predictor model is missing.")
    require((REPO_ROOT / "artifact/smoke_model/config.json").is_file(),
            "The smoke-test model metadata is missing.")

    if args.mode == "gpu":
        import torch
        require(torch.cuda.is_available(),
                "PyTorch cannot see a CUDA device; run --mode smoke instead.")
        require(torch.cuda.device_count() >= 1,
                "No CUDA devices are visible to this process.")
        print(f"CUDA devices visible: {torch.cuda.device_count()}")

    print(f"TeDiServe environment check PASS ({args.mode} mode).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TeDiServe environment check FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
