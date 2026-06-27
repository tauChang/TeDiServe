#!/usr/bin/env python3

import json
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from vllm.v1.resource_manager.reconfig_planner.milp_reconfig_planner import MILPReconfigPlanner
from vllm.v1.resource_manager.workload_monitor import WorkloadClass


# ============================================================
# 1. Load workload trace (ignore historical configs)
# ============================================================

def load_workload_trace(path):
    """
    Reads a JSON file containing entries like:
    [
      { "timestamp": "...",
        "config": {...},
        "workload_classes": [ {...}, ... ]
      },
      ...
    ]
    Returns:
        rps_trace: list[dict], where dict keys are "P_O" strings
    """
    with open(path) as f:
        raw = json.load(f)

    rps_trace = []
    for record in raw:
        wl = record["workload_classes"]

        rps_dict = {}
        for w in wl:
            p = w["prompt_length"]
            o = w["output_length"]
            key = f"{p}_{o}"
            rps_dict[key] = w["rps"]

        rps_trace.append(rps_dict)

    return rps_trace


def build_P_O_SLO_from_trace(rps_trace, default_slo=10.0):
    """
    Build dictionaries:
        P_k, O_k, SLO_k
    based on all prompt/output pairs in the workload trace.
    """
    P_k = {}
    O_k = {}
    SLO_k = {}

    for rps_dict in rps_trace:
        for key in rps_dict:
            p, o = map(int, key.split("_"))
            P_k[key] = p
            O_k[key] = o
            SLO_k[key] = default_slo

    return P_k, O_k, SLO_k


# ============================================================
# 2. Summaries and plotting (same as your existing code)
# ============================================================

def summarize_configurations(configs_over_time):
    records = []
    for t, cfg in configs_over_time:
        tp_counts = defaultdict(int)
        for _, bundles in cfg.items():
            tp = len(bundles)
            tp_counts[tp] += 1

        row = {"time": t}
        for tp, count in tp_counts.items():
            row[f"TP{tp}"] = count
        records.append(row)

    df = pd.DataFrame(records).fillna(0).sort_values("time")
    return df


def plot_reconfig_timeline(df, rps_trace, title="Reconfiguration Timeline"):
    tp_cols = [col for col in ["TP4", "TP2", "TP1"] if col in df.columns]
    time_steps = df["time"].to_numpy()
    total_rps = [sum(rps.values()) for rps in rps_trace]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # Top: RPS
    ax_top.plot(
        time_steps, total_rps, "k--o", linewidth=2.5,
        label="Total RPS", markersize=6
    )
    ax_top.set_ylabel("RPS")
    ax_top.set_title(title)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend()

    # Bottom: stacked TP bar chart
    bar_width = 0.6
    tp_colors = {"TP1": "#1f77b4", "TP2": "#2ca02c", "TP4": "#d62728"}

    bottom = np.zeros(len(time_steps))
    for tp in tp_cols:
        ax_bottom.bar(
            time_steps, df[tp], bottom=bottom,
            width=bar_width, label=tp,
            color=tp_colors[tp], edgecolor="black", linewidth=0.8
        )
        bottom += df[tp].to_numpy()

    ax_bottom.set_ylabel("Executors")
    ax_bottom.set_xlabel("Time step")
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bottom.legend()

    plt.xticks(time_steps)
    plt.tight_layout()
    plt.savefig("reconfig_timeline_from_trace.png", dpi=200)
    print("Saved plot: reconfig_timeline_from_trace.png")


# ============================================================
# 3. Core simulation loop (unchanged)
# ============================================================

async def simulate_reconfiguration(
    planner,
    node_to_bundles,
    current_config,
    rps_trace,
    P_k, O_k, SLO_k
):
    configs_over_time = []

    for t, RPS_k in enumerate(rps_trace):
        print(f"\n=== Time step {t} ===")
        print(f"Incoming RPS: {RPS_k}")

        # Convert into WorkloadClass list
        workload_classes = [
            WorkloadClass(
                prompt_length=P_k[k],
                output_length=O_k[k],
                slo=SLO_k[k],
                rps=RPS_k[k],
            )
            for k in RPS_k
        ]

        # Run MILP
        new_config = await planner.plan_reconfiguration_async(
            node_to_bundles=node_to_bundles,
            current_config=current_config,
            workload_classes=workload_classes,
            compare_with_fixed=False
        )

        print("→ New config:", new_config)
        configs_over_time.append((t, new_config))

        current_config = new_config

    return configs_over_time


# ============================================================
# 4. MAIN
# ============================================================

async def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--trace", required=True, help="Path to workload trace JSON")
    # args = parser.parse_args()
    trace_file = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/152122/config_history.json"

    # -------------------------
    # Load real workload trace
    # -------------------------
    rps_trace = load_workload_trace(trace_file)
    print(f"Loaded {len(rps_trace)} timesteps from workload trace")

    # Dynamically build P_k, O_k, SLO_k
    P_k, O_k, SLO_k = build_P_O_SLO_from_trace(rps_trace, default_slo=10.0)

    # -------------------------
    # YOU SUPPLY INITIAL CONFIG
    # -------------------------
    current_config = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
        2: [8, 9, 10, 11],
        3: [12, 13, 14, 15],
    }

    num_instances = len(current_config)
    node_to_bundles = {f"node{i}": [i * 4 + j for j in range(4)]
                       for i in range(num_instances)}

    # -------------------------
    # Planner
    # -------------------------
    latency_profile_paths = {
        # 1: "/work2/10446/tchang85/stampede3/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/H100/TP1.json",
        # 2: "/work2/10446/tchang85/stampede3/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/H100/TP2.json",
        # 4: "/work2/10446/tchang85/stampede3/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/H100/TP4.json",
        1: "/u/tchang85/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP1.json",
        2: "/u/tchang85/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP2.json",
        4: "/u/tchang85/dllm/latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP4.json",
    }

    planner = MILPReconfigPlanner(
        latency_profile_paths=latency_profile_paths,
        cache_prefix=True,
        cache_suffix=True,
        denoise_block_size=32
    )

    # -------------------------
    # Run simulation on trace
    # -------------------------
    configs_over_time = await simulate_reconfiguration(
        planner,
        node_to_bundles,
        current_config,
        rps_trace,
        P_k, O_k, SLO_k
    )

    # Print summary
    print("\n=== Summary ===")
    for t, cfg in configs_over_time:
        print(f"t={t}: {cfg}")

    df_summary = summarize_configurations(configs_over_time)
    print(df_summary)

    # Plot
    plot_reconfig_timeline(df_summary, rps_trace)


# ============================================================

if __name__ == "__main__":
    asyncio.run(main())
