import time
# from vllm.v1.resource_manager.reconfig_planner.milp_reconfig_planner_bak_1116 import MILPReconfigPlanner
from vllm.v1.resource_manager.reconfig_planner.milp_reconfig_planner import MILPReconfigPlanner
from vllm.v1.resource_manager.workload_monitor import WorkloadClass

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

import numpy as np


def summarize_configurations(configs_over_time):
    """
    Convert configuration history into a table:
    rows = timesteps, columns = TP degrees, values = number of executors
    """
    records = []
    for t, cfg in configs_over_time:
        # Count executors by TP degree (number of bundles per executor)
        tp_counts = defaultdict(int)
        for _, bundles in cfg.items():
            tp = len(bundles)
            tp_counts[tp] += 1

        record = {"time": t}
        for tp, count in tp_counts.items():
            record[f"TP{tp}"] = count
        records.append(record)

    df = pd.DataFrame(records).fillna(0).sort_values("time")
    return df


import matplotlib.pyplot as plt
import numpy as np

def plot_reconfig_timeline(df, rps_trace, title="Reconfiguration Timeline"):
    """
    Two-subplot visualization:
      • Upper: total and per-class RPS over time
      • Lower: stacked bars showing executor configuration by TP degree
    Shared x-axis (time steps).
    Always stacks TP=4, TP=2, TP=1 in that order (bottom→top).
    """
    # --- Prepare data ---
    tp_cols = [col for col in ["TP4", "TP2", "TP1"] if col in df.columns]
    time_steps = df["time"].to_numpy()
    total_rps = [sum(rps.values()) for rps in rps_trace]

    # --- Setup figure ---
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # =====================================================
    # 1️⃣  Top subplot: RPS over time
    # =====================================================
    class_names = list(rps_trace[0].keys())
    colors_rps = plt.cm.tab10.colors  # distinct qualitative colors

    # for i, cname in enumerate(class_names):
    #     series = [rps[cname] for rps in rps_trace]
    #     ax_top.plot(
    #         time_steps,
    #         series,
    #         "-o",
    #         label=cname,
    #         color=colors_rps[i % len(colors_rps)],
    #         linewidth=2,
    #     )

    ax_top.plot(
        time_steps,
        total_rps,
        "k--o",
        linewidth=2.5,
        label="Total RPS",
        markersize=6,
    )
    ax_top.set_ylabel("RPS", fontsize=11)
    ax_top.set_title(title, fontsize=13)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    ax_top.legend(fontsize=9, ncol=2, loc="upper left")

    # =====================================================
    # 2️⃣  Bottom subplot: Executor configuration (stacked bars)
    # =====================================================
    bar_width = 0.6
    tp_colors = {
        "TP1": "#1f77b4",  # blue
        "TP2": "#2ca02c",  # green
        "TP4": "#d62728",  # red
    }

    bottom = np.zeros(len(time_steps))
    for tp in tp_cols:  # enforce stacking order TP4→TP2→TP1
        ax_bottom.bar(
            time_steps,
            df[tp],
            bottom=bottom,
            width=bar_width,
            label=tp,
            color=tp_colors[tp],
            edgecolor="black",
            linewidth=0.8,
        )
        bottom += df[tp].to_numpy()

    ax_bottom.set_ylabel("Executors", fontsize=11)
    ax_bottom.set_xlabel("Time step", fontsize=11)
    ax_bottom.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax_bottom.legend(fontsize=9, loc="upper left")

    # --- Formatting ---
    plt.xticks(time_steps)
    plt.tight_layout()
    plt.savefig("reconfig_timeline_subplots_prefix.png", dpi=200)


async def simulate_reconfiguration(planner, node_to_bundles, current_config, rps_trace, P_k, O_k, SLO_k):
    """
    Simulate dynamic workload changes over time.
    Args:
        planner: MILPReconfigPlanner instance
        node_to_bundles: {node: [bundle_ids]}
        current_config: initial executor→bundle mapping
        rps_trace: list[dict[str, float]], time series of RPS per class
        P_k, O_k, SLO_k: dicts defining workload properties
    """
    configs_over_time = []

    for t, RPS_k in enumerate(rps_trace):
        print(f"\n=== Time step {t} ===")
        print(f"Incoming RPS: {RPS_k}")

        # Build workload classes for this timestep
        workload_classes = [
            WorkloadClass(
                prompt_length=P_k[k],
                output_length=O_k[k],
                slo=SLO_k[k],
                rps=RPS_k[k],
            )
            for k in RPS_k.keys()
        ]

        # Run planner
        new_config = await planner.plan_reconfiguration_async(
            node_to_bundles=node_to_bundles,
            current_config=current_config,
            workload_classes=workload_classes,
        )

        print("→ New config:", new_config)
        configs_over_time.append((t, new_config))

        # Update current config for next timestep
        current_config = new_config

        # (Optional) small sleep to simulate wall-clock delay
        # time.sleep(1)

    return configs_over_time


async def main():
    latency_profile_paths = {
        1: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP1.json",
        2: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP2.json",
        4: "./latency_profiles/GSAI-ML_LLaDA-8B-Instruct/GH200/TP4.json",
    }

    # K = ["1024_256", "1280_256", "1536_256"]
    # K = ["1024_256"]
    # P_k = {"1024_256": 1024, "1280_256": 1280, "1536_256": 1536}
    # O_k = {"1024_256": 256, "1280_256": 256, "1536_256": 256}
    # SLO_k = {"1024_256": 5.0, "1280_256": 5.0, "1536_256": 5.0}
    K = [(768, 256), (1024, 256), (1280, 256)]
    P_k = {f"{p}_{o}": p for p, o in K}
    O_k = {f"{p}_{o}": o for p, o in K}
    SLO_k = {f"{p}_{o}": 5.0 for p, o in K}
    RPS_k = None
    K = [f"{p}_{o}" for p, o in K]

    # Node layout
    num_instances = 2
    node_to_bundles = {f"node{i}": [i * 4 + j for j in range(4)] for i in range(num_instances)}
    current_config = {
        0: [0, 1, 2, 3], 
        1: [4, 5],
        2: [6],
        3: [7],
        # 2: [8, 9, 10, 11],
        # 3: [12, 13, 14, 15],
        # 4: [16, 17, 18, 19],
        # 5: [20, 21, 22, 23],
        # 6: [24, 25, 26, 27],
        # 7: [28, 29, 30, 31],
    }

    # Example RPS trace (time series)
    # rps_trace = [
    #     {"1024_256": 0, "1280_256": 0, "1536_256": 2},
    #     {"1024_256": 0, "1280_256": 2, "1536_256": 0},
    #     {"1024_256": 2, "1280_256": 0, "1536_256": 0},
    #     # {"1024_256": .4, "1280_256": .4, "1536_256": 0.2},
    # ]
    # rps_trace = [
    #     {"1024_256": 2.3},
    #     # {"1280_256": 1.6*5},
    #     # {"1536_256": 1.6*5},
    #     # {"1024_256": .4, "1280_256": .4, "1536_256": 0.2},
    # ]
    # rps_trace = [
    #     {"1024_256": 0.1, "1280_256": 0.06666666666666667},
    #     {"768_256": 0.1, "1024_256": 0.13333333333333333, "1280_256": 0.1},
    #     {"1024_256": 0.16666666666666666, "1280_256": 0.2},
    #     # {"1024_256": 0.1, "1280_256": 0.16666},
    # ]

    rps_trace = []
    # rps_classless_trace = [.4, .8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    rps_classless_trace = [1/1, 1/0.5]
    # rps_classless_trace = [1/0.25]
    # rps_classless_trace += rps_classless_trace[-2::-1]  # ramp down
    # rps_trace = rps_classless_trace
    # # rps_classless_trace = [6.0]
    # rps_classless_trace = [0.1 * i for i in range(1, 60)]  # 0.1 to 2.0
    # rps_trace = []
    for rps in rps_classless_trace:
        rps_trace.append({k: rps / len(K) for k in K})

    planner = MILPReconfigPlanner(
        latency_profile_paths=latency_profile_paths,
        cache_prefix=True,
        cache_suffix=True,
        denoise_block_size=32)

    configs_over_time = await simulate_reconfiguration(
        planner,
        node_to_bundles=node_to_bundles,
        current_config=current_config,
        rps_trace=rps_trace,
        P_k=P_k,
        O_k=O_k,
        SLO_k=SLO_k,
    )

    # --- Optional: visualize evolution ---
    print("\n=== Summary: configuration evolution ===")
    for t, cfg in configs_over_time:
        print(f"t={t}: {cfg}")
    
    df_summary = summarize_configurations(configs_over_time)
    print("\nExecutor timeline summary:\n", df_summary)

    # Plot
    plot_reconfig_timeline(df_summary, rps_trace)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())