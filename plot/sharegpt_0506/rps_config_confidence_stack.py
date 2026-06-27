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
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 22,
        "lines.linewidth": 4.5,
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


def resolve_benchmark_latency_info(path):
    if path.endswith("benchmark_latency_info.json"):
        return path
    return os.path.join(path, "logs", "benchmark_latency_info.json")

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

    tp_counts = {1: [], 2: [], 4: []}
    for e in trace:
        c = e["config"]
        counts = {1: 0, 2: 0, 4: 0}
        for workers in c.values():
            tp = len(workers)
            if tp in counts:
                counts[tp] += 1
        for tp, values in tp_counts.items():
            values.append(counts[tp])

    data = {"time": rel_times}
    for tp, values in tp_counts.items():
        if any(value > 0 for value in values):
            data[f"TP-{tp}"] = values

    return pd.DataFrame(data)

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
def compute_confidence_timeline(cache_path, workload_file, bucket_seconds=1):
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
# SLO ATTAINMENT TIMELINE
# ======================================================================
def compute_slo_attainment_timeline(benchmark_latency_info_path,
                                    slo_seconds=12.5,
                                    bucket_seconds=60):
    benchmark_latency_info_path = resolve_benchmark_latency_info(
        benchmark_latency_info_path)

    with open(benchmark_latency_info_path) as f:
        data = json.load(f)

    root = data[0] if isinstance(data, list) else data
    instances = root["instances"] if "instances" in root else root

    records = []
    for req in instances.values():
        if not isinstance(req, dict):
            continue
        arrival_time = req.get("arrival_time")
        request_latency = req.get("request_latency")
        status = req.get("status")
        if arrival_time is None or request_latency is None or status is None:
            continue

        records.append((
            dt_to_ns(parse_ts(arrival_time)),
            status == "success" and float(request_latency) < slo_seconds,
        ))

    if not records:
        return np.array([]), np.array([])

    records.sort(key=lambda item: item[0])
    timestamps_ns = np.array([record[0] for record in records], dtype=np.int64)
    attained = np.array([record[1] for record in records], dtype=np.int64)

    bucketed = np.array([bucket_ns(t, bucket_seconds) for t in timestamps_ns],
                        dtype=np.int64)
    buckets, inv = np.unique(bucketed, return_inverse=True)
    
    total_count = np.bincount(inv)
    attained_count = np.bincount(inv, weights=attained)
    attainment = attained_count / total_count * 100.0

    t0 = buckets[0]
    times_min = (buckets - t0) / 1e9 / 60.0
    return times_min, attainment

# ======================================================================
# FINAL PLOT (THREE SUBPLOTS)
# ======================================================================
def plot_full_figure(qps_time, qps_vals, cfg_df,
                     conf_time, avg_conf, avg_maxc,
                     slo_timelines):
    fig, axs = plt.subplots(
        4, 1, figsize=(14, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]}
    )
    plt.subplots_adjust(hspace=0.15)

    # ------------------------------------------------------
    # TOP: QPS
    # ------------------------------------------------------
    axs[0].plot(qps_time, qps_vals, color="black")
    axs[0].set_ylabel("RPS")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    # set y ticks at 0, 10, 20
    axs[0].set_yticks([0, 10, 20])

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

    tp_colors = {"TP-1": "#FFCF71", "TP-2": "#B6771D", "TP-4": "#7B542F"}
    bottom = np.zeros(len(cfg_time))

    for tp in ["TP-4", "TP-2", "TP-1"]:
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
            label=tp,
            edgecolor=None
        )
        bottom = y_top

    axs[1].set_ylabel("# Instances", fontsize=22)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].set_ylim(0, bottom.max() * 1.1)
    # set y ticks at 0, 5, 10, 15
    axs[1].set_yticks(np.arange(0, bottom.max() * 1.1, 5))
    # axs[1].legend(loc="upper left")
    handles, labels = axs[1].get_legend_handles_labels()

    order = ["TP-1", "TP-2", "TP-4"]
    present_order = [tp for tp in order if tp in labels]
    if present_order:
        sorted_handles = [handles[labels.index(tp)] for tp in present_order]
        sorted_labels = present_order

        axs[1].legend(
            sorted_handles,
            sorted_labels,
            # bbox_to_anchor=(0.175, 1.02),
            labelspacing=0.13,
            borderpad=0.2,
            handletextpad=0.4,
            handlelength=1.3,
            loc="lower right"
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
    axs[2].set_ylabel("Confidence", fontsize=22)
    # axs[2].set_xlabel("Time (minutes)")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(handles,
                  ["Max Confidence Threshold", "Confidence Threshold Used"],
                  labelspacing=0.13,
                  borderpad=0.2,
                  handlelength=1,
                  handletextpad=0.4,
                  loc="lower right")
    axs[2].set_ylim(0.6, 0.92)
    axs[2].set_xlim(0, 120)
    # ytick at 0.9, 0.8, 0.7
    axs[2].set_yticks([0.6, 0.7, 0.8, 0.9])

    # ------------------------------------------------------
    # FOURTH: SLO ATTAINMENT
    # ------------------------------------------------------
    slo_colors = {
        "InFaaS": "#134686",
        "Llumnix": "#B89000",
        "TeDiServe": "#FF4F0F",
    }
    slo_order = ["InFaaS", "Llumnix", "TeDiServe"]
    for name in slo_order:
        if name not in slo_timelines:
            continue
        times_min, attainment = slo_timelines[name]
        axs[3].plot(times_min,
                    attainment,
                    label=name,
                    color=slo_colors.get(name))

    axs[3].set_ylabel("SLO Att. (%)", fontsize=22)
    axs[3].set_xlabel("Time (minutes)")
    axs[3].grid(True, linestyle="--", alpha=0.5)
    handles, labels = axs[3].get_legend_handles_labels()
    present_order = [name for name in slo_order if name in labels]
    sorted_handles = [handles[labels.index(name)] for name in present_order]
    axs[3].legend(sorted_handles,
                  present_order,
                  labelspacing=0.13,
                  borderpad=0.2,
                  handlelength=1,
                  handletextpad=0.4,
                  loc="lower left")
    axs[3].set_ylim(0, 110)
    axs[3].set_xlim(0, 120)
    axs[3].set_yticks([0, 25, 50, 75, 100])

    # plt.tight_layout()
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
    # tedi_base = "/u/tchang85/dllm/sbatch_experiment_dir/20260507/011626_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/"
    tedi_base = "/u/tchang85/dllm/sbatch_experiment_dir/20260506/114447_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/"
    config_json = tedi_base + "config_history.json"
    arrival_file = "/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_20qps.txt"

    step_data_file = tedi_base + "step_data.json"
    workload_history_file = tedi_base + "workload_history.json"
    slo_seconds = 12.5
    slo_baselines = {
        "TeDiServe": tedi_base + "logs/benchmark_latency_info.json",
        "Llumnix": "/u/tchang85/dllm/sbatch_experiment_dir/20260503/163209_sharegpt_20_llumnix_only/0_llumnix_20qps/",
        "InFaaS": "/u/tchang85/dllm/sbatch_experiment_dir/20260503/193754_sharegpt_20_infaas_only/0_infaas_20qps/",
    }

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
    bucket_seconds = 180
    conf_time, avg_conf, avg_maxc, active_vals = \
        compute_confidence_timeline(
            cache_path,
            workload_history_file,
            bucket_seconds=bucket_seconds
        )

    slo_timelines = {
        name: compute_slo_attainment_timeline(path,
                                              slo_seconds=slo_seconds,
                                              bucket_seconds=bucket_seconds)
        for name, path in slo_baselines.items()
    }

    # ---------------------------------------------------------
    # FINAL COMBINED PLOT
    # ---------------------------------------------------------
    plot_full_figure(qps_time, qps_vals, cfg_df,
                     conf_time, avg_conf, avg_maxc,
                     slo_timelines)
