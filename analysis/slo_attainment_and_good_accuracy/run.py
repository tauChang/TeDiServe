#!/usr/bin/env python3
import json
import pandas as pd
import argparse


def analyze_eval_results(path: str, slo: float = 8.0):
    """Load and analyze vLLM evaluation results from a JSON file."""
    # --- Load JSON -----------------------------------------------------------
    with open(path, "r") as f:
        data = json.load(f)

    # --- Flatten response times ---------------------------------------------
    response_times = pd.DataFrame([
        {"doc_id": int(k), "response_time": v}
        for k, v in data["response_times"].items()
    ])

    # --- Flatten samples -----------------------------------------------------
    task_name = next(iter(data["samples"].keys()))  # e.g., "gsm8k"
    samples = pd.DataFrame(data["samples"][task_name])
    samples = samples[samples["filter"] == "flexible-extract"].copy()
    samples["doc_id"] = samples["doc_id"].astype(int)

    # --- Merge and compute fields -------------------------------------------
    df = pd.merge(samples, response_times, on="doc_id", how="left")
    df["is_correct"] = df["exact_match"].astype(bool)

    # --- Compute metrics -----------------------------------------------------
    total = len(df)
    slo_met = df[df["response_time"] <= slo]
    slo_miss = df[df["response_time"] > slo]

    print(f"\nFile: {path}")
    print(f"Total flexible-extract samples: {total}")
    print(f"SLO attainment (≤{slo}s): {len(slo_met) / total:.3f}")
    print(f"Overall accuracy: {df['is_correct'].mean():.3f}")
    print(f"Average response time: {df['response_time'].mean():.3f}s")

    print(f"\n≤{slo}s:")
    print(f"  Count: {len(slo_met)}")
    print(f"  Accuracy: {slo_met['is_correct'].mean():.3f}")

    print(f">\n{slo}s:")
    print(f"  Count: {len(slo_miss)}")
    print(f"  Accuracy: {slo_miss['is_correct'].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vLLM evaluation JSON results.")
    parser.add_argument(
        "--path",
        type=str,
        default="/u/tchang85/dllm/experiment_dir/20251113_093505/results/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
        help="Path to the JSON results file.",
    )
    parser.add_argument("--slo", type=float, default=8.0, help="SLO in seconds.")
    args = parser.parse_args()

    analyze_eval_results(args.path, slo=args.slo)
