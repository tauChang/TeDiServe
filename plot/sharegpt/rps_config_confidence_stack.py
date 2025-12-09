#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bisect
from datetime import datetime
import os

# ======================================================================
# STYLING
# ======================================================================
plt.rcParams.update({
        "font.family": "serif",
        "font.size": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 26,
        "lines.linewidth": 3.5,
    })
    

# ======================================================================
# HELPERS
# ======================================================================
def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")

def dt_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)

def bucket_ns(t_ns: int, resolution_s: int) -> int:
    res = int(resolution_s * 1e9)
    return (t_ns // res) * res

# ======================================================================
# QPS FUNCTIONS
# ======================================================================
def load_arrivals(arrival_file):
    with open(arrival_file) as f:
        vals = [float(x.strip()) for x in f if x.strip()]
    return np.array(vals)

def compute_qps_from_arrivals(arrivals, window_minutes):
    arrivals = arrivals - arrivals.min()
    window_sec = window_minutes * 60

    T = arrivals.max()
    num_windows = int(np.ceil(T / window_sec))
    edges = np.arange(0, (num_windows + 1) * window_sec, window_sec)
    counts, _ = np.histogram(arrivals, bins=edges)
    qps = counts / window_sec
    qps_time = np.arange(len(qps)) * window_minutes
    return qps_time, qps

# ======================================================================
# CONFIG HISTORY
# ======================================================================
def load_reconfig_trace(json_file):
    trace = json.load(open(json_file))
    trace = trace[:-1] if len(trace) > 1 else trace

    timestamps = [datetime.fromisoformat(e["timestamp"]) for e in trace]
    t0 = timestamps[0]
    rel_times = [(t - t0).total_seconds() / 60. for t in timestamps]

    tp1, tp2, tp4 = [], [], []
    for e in trace:
        c = e["config"]
        c1 = c2 = c4 = 0
        for workers in c.values():
            tp = len(workers)
            if tp == 1: c1 += 1
            elif tp == 2: c2 += 1
            elif tp == 4: c4 += 1
        tp1.append(c1)
        tp2.append(c2)
        tp4.append(c4)

    return pd.DataFrame({"time": rel_times, "TP1": tp1, "TP2": tp2, "TP4": tp4})

# ======================================================================
# RAW CACHE BUILDER FOR CONFIDENCE
# ======================================================================
def build_raw_cache(step_data_path):
    cache_path = step_data_path + ".raw_cache.npz"
    if os.path.exists(cache_path):
        print("Using existing confidence cache:", cache_path)
        return cache_path

    print("Building confidence cache (first time only)…")
    ts_list, conf_list, max_list = [], [], []

    with open(step_data_path, "r") as f:
        for line in f:
            j = json.loads(line)
            rid = j.get("id")
            if "warmup" in str(rid).lower():
                continue
            c = j.get("confidence_threshold")
            m = j.get("max_confidence_threshold")
            if c is None or m is None:
                continue
            ts_list.append(dt_to_ns(parse_ts(j["timestamp"])))
            conf_list.append(c)
            max_list.append(m)

    np.savez(cache_path,
             timestamps_ns=np.array(ts_list, np.int64),
             conf=np.array(conf_list, np.float32),
             max_conf=np.array(max_list, np.float32))
    print("Cache saved:", cache_path)
    return cache_path

# ======================================================================
# ACTIVE REQUEST EVENTS
# ======================================================================
def build_active_events(workload_file):
    arrival, completion = {}, {}
    with open(workload_file, "r") as f:
        for line in f:
            j = json.loads(line)
            rid = j["request_id"]
            if "arrival_time" in j:
                arrival[rid] = dt_to_ns(parse_ts(j["arrival_time"]))
            if "completion_time" in j:
                completion[rid] = dt_to_ns(parse_ts(j["completion_time"]))

    events = []
    for rid in arrival:
        events.append((arrival[rid], +1))
    for rid in completion:
        events.append((completion[rid], -1))

    events.sort()
    times, deltas = zip(*events)
    prefix = []
    cur = 0
    for d in deltas:
        cur += d
        prefix.append(cur)
    return np.array(times), np.array(prefix)

def active_at(t_ns, times_ns, prefix):
    idx = bisect.bisect_right(times_ns, t_ns) - 1
    return prefix[idx] if idx >= 0 else 0

# ======================================================================
# BUILD CONFIDENCE TIMELINE (BOTTOM GRAPH)
# ======================================================================
def compute_confidence_timeline(cache_path, workload_file, bucket_seconds=60):
    cache = np.load(cache_path)
    ts = cache["timestamps_ns"]
    conf = cache["conf"]
    maxc = cache["max_conf"]

    # bucket
    bucketed = np.array([bucket_ns(t, bucket_seconds) for t in ts], np.int64)

    buckets, inv = np.unique(bucketed, return_inverse=True)
    sum_conf = np.bincount(inv, weights=conf)
    sum_maxc = np.bincount(inv, weights=maxc)
    count = np.bincount(inv)

    avg_conf = sum_conf / count
    avg_maxc = sum_maxc / count

    # active requests
    times_ns, prefix = build_active_events(workload_file)
    active_vals = [active_at(t, times_ns, prefix) for t in buckets]

    # convert to minutes from start
    t0 = buckets[0]
    times_min = (buckets - t0) / 1e9 / 60.0

    return times_min, avg_conf, avg_maxc, active_vals

# ======================================================================
# FINAL PLOT (THREE SUBPLOTS)
# ======================================================================
def plot_full_figure(qps_time, qps_vals, cfg_df,
                     conf_time, avg_conf, avg_maxc):
    fig, axs = plt.subplots(
        3, 1, figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3, 1.2]}
    )

    # ------------------------------------------------------
    # TOP: QPS
    # ------------------------------------------------------
    axs[0].plot(qps_time, qps_vals, color="black")
    axs[0].set_ylabel("RPS")
    axs[0].grid(True, linestyle="--", alpha=0.5)

    # ------------------------------------------------------
    # MIDDLE: CONFIG
    # ------------------------------------------------------
    cfg_time = cfg_df["time"].to_numpy()
    df = cfg_df

    if len(cfg_time) > 1:
        dt = cfg_time[-1] - cfg_time[-2]
    else:
        dt = 1
    edges = np.append(cfg_time, cfg_time[-1] + dt)

    tp_colors = {"TP1": "#FFCF71", "TP2": "#B6771D", "TP4": "#7B542F"}
    bottom = np.zeros(len(cfg_time))

    for tp in ["TP4", "TP2", "TP1"]:
        if tp not in df.columns:
            continue
        y = df[tp].to_numpy()
        y_top = bottom + y

        axs[1].fill_between(
            edges,
            np.append(bottom, bottom[-1]),
            np.append(y_top, y_top[-1]),
            step="post",
            color=tp_colors[tp],
            alpha=0.85,
            label=tp
        )
        bottom = y_top

    axs[1].set_ylabel("# Model Instances")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].set_ylim(0, bottom.max() * 1.1)
    # axs[1].legend(loc="upper left")
    handles, labels = axs[1].get_legend_handles_labels()

    order = ["TP1", "TP2", "TP4"]
    sorted_handles = [handles[labels.index(tp)] for tp in order]
    sorted_labels  = order

    axs[1].legend(
        sorted_handles,
        sorted_labels,
        bbox_to_anchor=(0.15, 0.99),
        loc="upper left"
    )

    # ------------------------------------------------------
    # BOTTOM: CONFIDENCE TIMELINE
    # ------------------------------------------------------
    axs[2].plot(conf_time, avg_conf, label="Confidence Threshold Used", color="#4BB4AA")
    axs[2].plot(conf_time, avg_maxc, label="Max Confidence Threshold", color="#540863")
    handles = [
        axs[2].lines[1],
        axs[2].lines[0],
    ]
    axs[2].set_ylabel("Confidence")
    axs[2].set_xlabel("Time (minutes)")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(handles,
                  ["Max Confidence Threshold", "Confidence Threshold Used"],
                  loc="lower left")
    axs[2].set_ylim(0.6, 0.92)
    axs[2].set_xlim(0, 120)

    plt.tight_layout()
    plt.savefig("full_timeline.pdf", bbox_inches="tight")
    plt.savefig("full_timeline.png", dpi=300, bbox_inches="tight")
    print("Saved full_timeline.png")

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":

    # ---------------------------------------------------------
    # INPUT PATHS (CHANGE THESE)
    # ---------------------------------------------------------
    config_json = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/config_history.json"
    arrival_file = "/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_22qps.txt"

    step_data_file = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/step_data.json"
    workload_history_file = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/workload_history.json"

    # ---------------------------------------------------------
    # LOAD QPS + CONFIG
    # ---------------------------------------------------------
    cfg_df = load_reconfig_trace(config_json)
    qps_time, qps_vals = compute_qps_from_arrivals(
        load_arrivals(arrival_file), 1.0
    )

    # ---------------------------------------------------------
    # BUILD CONFIDENCE CACHE
    # ---------------------------------------------------------
    cache_path = build_raw_cache(step_data_file)

    # ---------------------------------------------------------
    # COMPUTE CONFIDENCE TIMELINE
    # ---------------------------------------------------------
    conf_time, avg_conf, avg_maxc, active_vals = \
        compute_confidence_timeline(
            cache_path,
            workload_history_file,
            bucket_seconds=60
        )

    # ---------------------------------------------------------
    # FINAL COMBINED PLOT
    # ---------------------------------------------------------
    plot_full_figure(qps_time, qps_vals, cfg_df,
                     conf_time, avg_conf, avg_maxc)
