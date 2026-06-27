#!/usr/bin/env python3
import json
import os
import numpy as np
from datetime import datetime
import bisect
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")

def datetime_to_ns(dt: datetime) -> int:
    """Convert datetime to integer nanoseconds."""
    return int(dt.timestamp() * 1e9)

def bucket_ns(t_ns: int, resolution_s: int) -> int:
    """Bucket timestamp in nanoseconds to any resolution in seconds."""
    res_ns = int(resolution_s * 1e9)
    return (t_ns // res_ns) * res_ns

# -------------------------------------------------------------
# One-time cache builder
# -------------------------------------------------------------

def build_raw_cache(step_data_path):
    """
    Stream the huge JSON file once and build a compact binary cache
    containing only timestamps + conf + max_conf.
    """
    cache_path = step_data_path + ".raw_cache.npz"
    if os.path.exists(cache_path):
        print(f"Raw cache already exists: {cache_path}")
        return cache_path

    print(f"Building raw cache at {cache_path}...")
    ts_list = []
    conf_list = []
    maxconf_list = []

    with open(step_data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry.get("id")

            if "warmup" in str(rid).lower():
                continue

            # parse timestamp
            ts_ns = datetime_to_ns(parse_timestamp(entry["timestamp"]))
            conf = entry.get("confidence_threshold")
            maxc = entry.get("max_confidence_threshold")

            if conf is None or maxc is None:
                continue

            ts_list.append(ts_ns)
            conf_list.append(conf)
            maxconf_list.append(maxc)

    print("Converting to numpy arrays...")
    ts = np.array(ts_list, dtype=np.int64)
    conf = np.array(conf_list, dtype=np.float32)
    maxc = np.array(maxconf_list, dtype=np.float32)

    print("Saving cache...")
    np.savez(cache_path, timestamps_ns=ts, conf=conf, max_conf=maxc)
    print("Done.")

    return cache_path

# -------------------------------------------------------------
# Build event list for active requests
# -------------------------------------------------------------

def build_active_events(workload_history_path):
    arrival = {}
    completion = {}

    print("Loading workload_history.json …")
    with open(workload_history_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry["request_id"]

            if "arrival_time" in entry:
                arrival[rid] = datetime_to_ns(parse_timestamp(entry["arrival_time"]))
            if "completion_time" in entry:
                completion[rid] = datetime_to_ns(parse_timestamp(entry["completion_time"]))

    events = []
    for rid in arrival:
        events.append((arrival[rid], +1))
    for rid in completion:
        events.append((completion[rid], -1))

    events.sort()

    times_ns, deltas = zip(*events)

    prefix_active = []
    cur = 0
    for d in deltas:
        cur += d
        prefix_active.append(cur)

    return np.array(times_ns, dtype=np.int64), np.array(prefix_active, dtype=np.int32)

def active_at_time(t_ns, event_times_ns, prefix_active):
    idx = bisect.bisect_right(event_times_ns, t_ns) - 1
    return prefix_active[idx] if idx >= 0 else 0

# -------------------------------------------------------------
# Build global timeline for any resolution
# -------------------------------------------------------------

def build_global_timeline_from_cache(
    cache_path,
    workload_history_path,
    resolution_s,
    output_png
):
    print(f"Loading raw cache: {cache_path}")
    cache = np.load(cache_path)

    ts = cache["timestamps_ns"]
    conf = cache["conf"]
    maxc = cache["max_conf"]

    # 1. Bucket timestamps
    print(f"Bucketing timestamps at {resolution_s} seconds...")
    bucketed = np.array([bucket_ns(t, resolution_s) for t in ts], dtype=np.int64)

    unique_buckets, inverse_idx = np.unique(bucketed, return_inverse=True)

    # 2. Aggregate confidence / max confidence
    print("Aggregating confidence…")
    sum_conf = np.bincount(inverse_idx, weights=conf)
    sum_maxc = np.bincount(inverse_idx, weights=maxc)
    count = np.bincount(inverse_idx)

    avg_conf = sum_conf / count
    avg_maxc = sum_maxc / count

    # 3. Compute active requests for each bucket
    print("Computing active request counts…")
    event_times_ns, prefix_active = build_active_events(workload_history_path)
    active_vals = [
        active_at_time(t, event_times_ns, prefix_active)
        for t in unique_buckets
    ]

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    print("Plotting…")

    # Convert nanoseconds back to datetime
    bucket_times = [datetime.fromtimestamp(t / 1e9) for t in unique_buckets]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7))

    fig.suptitle(
        f"Global Confidence & Active Requests ({resolution_s}-second buckets)"
    )

    axs[0].plot(bucket_times, avg_conf, label="avg_conf", marker="o", markersize=2)
    axs[0].plot(bucket_times, avg_maxc, label="avg_max_conf", marker="x", markersize=3)
    axs[0].set_ylabel("Confidence")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].step(bucket_times, active_vals, where="post")
    axs[1].set_ylabel("# Active Requests")
    axs[1].set_xlabel("Time")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()

    print(f"Saved plot: {output_png}")

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main(step_data_path, workload_history_path, output_dir, resolution_s=60):
    os.makedirs(output_dir, exist_ok=True)

    # Build cache if not exists
    cache_path = build_raw_cache(step_data_path)

    # Build timeline for chosen resolution
    out_png = os.path.join(output_dir, f"global_conf_{resolution_s}s.png")
    build_global_timeline_from_cache(cache_path, workload_history_path, resolution_s, out_png)


if __name__ == "__main__":
    # MODIFY THESE PATHS
    step_data_file = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/step_data.json"
    workload_history_file = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130/203923/workload_history.json"
    output_directory = "./"

    # Choose bucket size (seconds)
    # Examples:
    #   1   -> 1-second buckets
    #   10  -> 10-second buckets
    #   60  -> 1-minute buckets
    #   300 -> 5-minute buckets
    resolution_s = 120

    main(step_data_file, workload_history_file, output_directory, resolution_s)
