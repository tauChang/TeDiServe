#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

plt.rcParams.update({
        "font.family": "serif",
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
    })

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#1f77b4", "#2ca02c", "#d62728"]   # or any hex colors you want
)

# ============================================================
# Load arrival timestamps → QPS curve
# ============================================================
def load_arrivals(arrival_file):
    with open(arrival_file) as f:
        vals = [float(x.strip()) for x in f if x.strip()]
    return np.array(vals)


def compute_qps_from_arrivals(arrivals, window_minutes):
    arrivals = arrivals - arrivals.min()
    window_sec = window_minutes * 60

    T = arrivals.max()
    num_windows = int(np.ceil(T / window_sec))

    window_edges = np.arange(0, (num_windows + 1) * window_sec, window_sec)

    counts, _ = np.histogram(arrivals, bins=window_edges)
    qps = counts / window_sec

    # Time axis in minutes (raw!)
    qps_time = np.arange(len(qps)) * window_minutes
    return qps_time, qps


# ============================================================
# Load config_history.json → DataFrame(df)
# ============================================================
def load_reconfig_trace(json_file):
    trace = json.load(open(json_file))

    # Ignore last entry
    if len(trace) > 1:
        trace = trace[:-1]

    timestamps = []
    tp1_list, tp2_list, tp4_list = [], [], []

    for entry in trace:
        ts = datetime.fromisoformat(entry["timestamp"])
        timestamps.append(ts)

    t0 = timestamps[0]
    rel_times = [(t - t0).total_seconds() / 60.0 for t in timestamps]

    for entry in trace:
        config = entry["config"]

        tp1 = tp2 = tp4 = 0
        for _, workers in config.items():
            tp = len(workers)
            if tp == 1:
                tp1 += 1
            elif tp == 2:
                tp2 += 1
            elif tp == 4:
                tp4 += 1

        tp1_list.append(tp1)
        tp2_list.append(tp2)
        tp4_list.append(tp4)

    df = pd.DataFrame({
        "time": rel_times,
        "TP1": tp1_list,
        "TP2": tp2_list,
        "TP4": tp4_list,
    })

    return df


# ============================================================
# Plot with shared x-axis (NO interpolation)
# ============================================================
def plot_reconfig_timeline(df, qps_time, qps_vals, title="Reconfig Timeline"):

    cfg_time = df["time"].to_numpy()

    # unified x-axis range
    xmin = 0
    # xmax = max(cfg_time.max(), qps_time.max())
    xmax = 120

    # step edges for config step-area plot
    if len(cfg_time) > 1:
        dt = cfg_time[-1] - cfg_time[-2]
    else:
        dt = 1.0
    time_edges = np.append(cfg_time, cfg_time[-1] + dt)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3]}
    )

    # -------------------------
    # TOP: RAW QPS curve
    # -------------------------
    ax_top.plot(
        qps_time,
        qps_vals,
        # "k--o",
        color="black",
        linewidth=2.0,
        markersize=5,
    )
    ax_top.set_ylabel("RPS")
    # ax_top.set_title(title, fontsize=15)
    ax_top.grid(True, linestyle="--", alpha=0.5)
    # ax_top.legend(loc="upper left")
    ax_top.set_xlim(xmin, xmax)
    # x tick every 10 minutes
    ax_top.set_xticks(np.arange(xmin, xmax + 1, 10))

    # -------------------------
    # BOTTOM: step area config
    # -------------------------
    tp_colors = {"TP1": "#1f77b4", "TP2": "#2ca02c", "TP4": "#d62728"}
    bottom = np.zeros(len(cfg_time))

    for tp in ["TP4", "TP2", "TP1"]:
        if tp not in df.columns:
            continue

        y = df[tp].to_numpy()
        y_top = bottom + y

        ax_bottom.fill_between(
            time_edges,
            np.append(bottom, bottom[-1]),
            np.append(y_top, y_top[-1]),
            step="post",
            color=tp_colors[tp],
            alpha=0.85,
            label=tp,
            linewidth=0.0,
        )

        bottom = y_top

    ax_bottom.set_ylabel("# Model Instances")
    ax_bottom.set_xlabel("Time (minutes from start)")
    ax_bottom.grid(True, linestyle="--", alpha=0.5)
    # ax_bottom.legend(loc="upper left")
    ax_bottom.legend(
        bbox_to_anchor=(0.14, 0.99), loc="upper left"
    )
    ax_bottom.set_xlim(xmax=xmax)
    ax_bottom.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig("reconfig_timeline_from_trace.png", dpi=200)
    print("Saved: reconfig_timeline_from_trace.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config_json = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/config_history.json"
    arrival_file = "/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_22qps.txt"

    df = load_reconfig_trace(config_json)

    qps_time, qps_vals = compute_qps_from_arrivals(
        load_arrivals(arrival_file),
        window_minutes=1.0,  # raw 1-min windows
    )
    print(df)
    print(qps_time)

    plot_reconfig_timeline(df, qps_time, qps_vals, title="Reconfig Timeline")
