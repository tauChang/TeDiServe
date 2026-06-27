#!/usr/bin/env python3
import json
import pandas as pd
import argparse
import numpy as np


# ==============================================================================
# Loading Helpers
# ==============================================================================

def load_arrivals(trace_path):
    """Load arrival timestamps from txt file (one float per line)."""
    arrivals = np.loadtxt(trace_path)
    return arrivals


def load_instances(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # data may be a list if you're appending results
    if isinstance(data, list):
        data = data[-1]   # take last entry

    instances = data["instances"]

    # Convert to DataFrame indexed by request ID (as string)
    df = pd.DataFrame.from_dict(instances, orient="index")

    if "status" not in df.columns:
        df["status"] = "success"

    # Convert index -> int request_id
    df.index = df.index.astype(int)
    df = df.sort_index()

    # Optional stats
    df["total_tokens"] = df["prompt_len"] + df["expected_response_len"]

    print("Latency stats:")
    print(df["request_latency"].describe())

    print("\nPrompt length stats:")
    print(df["prompt_len"].describe())

    print("\nExpected response length stats:")
    print(df["expected_response_len"].describe())

    return df


def analyze_eval_results(json_path, trace_path, slo=3.0, t_start=0.0, t_end=999999):
    df = load_instances(json_path)
    arrivals = load_arrivals(trace_path)

    if len(arrivals) < len(df):
        print(f"WARNING: trace has fewer arrivals ({len(arrivals)}) than requests ({len(df)})")

    # Attach arrival times to dataframe
    df["arrival_time"] = arrivals[:len(df)]
    # Filter based on time window
    mask = (df["arrival_time"] >= t_start) & (df["arrival_time"] < t_end)
    df_window = df[mask]

    print(f"\n=== Filtering requests arriving in [{t_start}, {t_end}) sec ===")
    print(f"Total requests in window: {len(df_window)}")

    if len(df_window) == 0:
        print("No requests in this time window.")
        return

    avg_latency = df_window["request_latency"].mean()
    print(f"Average latency (filtered): {avg_latency:.2f} sec")

    # Compute SLO attainment
    dropped_requests = (df_window["status"] == "dropped").sum()
    within_slo = (
        (df_window["status"] == "success") &
        (df_window["request_latency"] <= slo)
    ).sum()
    total = len(df_window)
    dropped_percentage = dropped_requests / total * 100.0
    slo_attainment = within_slo / total * 100.0

    print(f"Dropped requests: {dropped_percentage:.2f}% ({dropped_requests}/{total})")
    print(f"SLO Attainment (@ {slo} sec): {slo_attainment:.2f}% ({within_slo}/{total})")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    # path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251211/013824/logs/benchmark_latency_info.json"
    # path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251211/011511/logs/benchmark_latency_info.json"
    # path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251211/021826/logs/benchmark_latency_info.json"
    # trace = "/work2/10446/tchang85/stampede3/BurstGPT/arrival_trace_4_12_20.txt"
    paths = {
        "TeDi": "/u/tchang85/dllm/sbatch_experiment_dir/20260507/011626_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/logs/benchmark_latency_info.json",
        "Llumnix": "/u/tchang85/dllm/sbatch_experiment_dir/20260503/163209_sharegpt_20_llumnix_only/0_llumnix_20qps/logs/benchmark_latency_info.json",
        "InFaaS": "/u/tchang85/dllm/sbatch_experiment_dir/20260503/193754_sharegpt_20_infaas_only/0_infaas_20qps/logs/benchmark_latency_info.json",
    }
    trace = "/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_20qps.txt"
    slo = 12.5
    start_min = 45
    end_min = 120
    t_start = start_min * 60.0
    t_end = end_min * 60.0

    for name, path in paths.items():
        print(f"\n================ Analyzing {name} ===")

        analyze_eval_results(
            path,
            trace,
            slo=slo,
            t_start=t_start,
            t_end=t_end,
        )
