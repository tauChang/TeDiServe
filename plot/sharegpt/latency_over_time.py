import numpy as np
import matplotlib.pyplot as plt
import json
import os

BASE = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251130"
FILES = {
    "TeDiServe": "203923",
    "Llumnix": "231147",
    "InFaaS": "152122",
}
COLORS = {
    "InFaaS": "#134686",
    "Llumnix": "#E5C95F",
    "TeDiServe": "#FF4F0F",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 2.5,
})

# ============================================================
# Load arrival timestamps (flat list)
# ============================================================
def load_arrival_trace(path):
    with open(path) as f:
        vals = [float(x.strip()) for x in f if x.strip()]
    return np.array(vals, dtype=float)   # seconds since start


# ============================================================
# Load latencies + align with arrival times
# ============================================================
def load_latencies_with_arrival(path, arrival_times):
    """
    Returns:
      list of (arrival_sec, latency_sec)
    """
    with open(path, "r") as f:
        data = json.load(f)

    insts = data[0]["instances"]
    results = []

    for i, inst in insts.items():
        i = int(i)

        if i >= len(arrival_times):
            # shouldn't happen, but guard
            continue

        if "request_latency" not in inst:
            continue

        arr = arrival_times[i]
        lat = float(inst["request_latency"])
        results.append((arr, lat))

    return results


# ============================================================
# Bucket into fixed windows (e.g., 5-minute = 300 sec)
# ============================================================
def bucket_latency(records, bucket_seconds=300):
    """
    records: list of (arrival_sec, latency_sec)
    Return: dict bucket_index -> list_of_latencies
    """
    buckets = {}
    for (arr, lat) in records:
        b = int(arr // bucket_seconds)
        buckets.setdefault(b, []).append(lat)
    return buckets


# ============================================================
# Plot boxplot timeline
# ============================================================
def plot_latency_time_buckets(all_system_buckets,
                              bucket_seconds=300,
                              colors=COLORS,
                              outfile="latency_timeline.png"):
    """
    all_system_buckets: dict system -> dict bucket -> list(latencies)
    """

    systems = list(all_system_buckets.keys())

    all_buckets = sorted({
        b
        for sys in systems
        for b in all_system_buckets[sys].keys()
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    positions = []
    box_data = []
    box_colors = []

    # Offset for multiple systems in same bucket
    offset = 0.2
    num_sys = len(systems)

    for sys_idx, sys in enumerate(systems):
        buckets = all_system_buckets[sys]
        color = colors.get(sys, "black")

        for b in all_buckets:
            if b not in buckets:
                continue
            lats = buckets[b]
            if len(lats) == 0:
                continue

            # X-axis position = bucket_index + offset*(sys_idx - mid)
            shift = offset * (sys_idx - (num_sys-1)/2)
            pos = b + shift

            box_data.append(lats)
            positions.append(pos)
            box_colors.append(color)

        # print the average latency from 0 - 45 minutes, and 45 -- 120 for each system
        total_lats = []
        total_lats_45 = []
        for b in all_buckets:
            if b not in buckets:
                continue
            lats = buckets[b]
            total_lats.extend(lats)
            if b < 9:   # 9 * 5min = 45min
                total_lats_45.extend(lats)
        avg_lat = np.mean(total_lats) if len(total_lats) > 0 else float('nan')
        avg_lat_45 = np.mean(total_lats_45) if len(total_lats_45) > 0 else float('nan')
        avg_lat_later = np.mean([lat for lat in total_lats if lat not in total_lats_45]) if len(total_lats) > len(total_lats_45) else float('nan')
        print(f"{sys} average latency 0-45min: {avg_lat_45:.2f} sec, 45-120min: {avg_lat_later:.2f} sec, overall: {avg_lat:.2f} sec")
        

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.15,
        patch_artist=True,
        showfliers=False,
    )

    # color each box
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xlabel("Time (minutes from start)")
    ax.set_ylabel("Latency (sec)")
    ax.set_title("Latency Over Time (5-min buckets)")

    # bucket index → minutes
    xticks = all_buckets
    xtick_labels = [f"{b * bucket_seconds / 60:.0f}" for b in all_buckets]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=220)
    print(f"Saved {outfile}")

def main():
    arrival_trace_path = "/work2/10446/tchang85/stampede3/BurstGPT/burstgpt_2hrs_22qps.txt"

    arrival_times = load_arrival_trace(arrival_trace_path)

    all_system_buckets = {}

    for system, ts in FILES.items():
        json_path = os.path.join(BASE, ts, "logs", "benchmark_latency_info.json")
        print(f"Loading {system}")
        recs = load_latencies_with_arrival(json_path, arrival_times)
        buckets = bucket_latency(recs, bucket_seconds=300)   # 5 min
        all_system_buckets[system] = buckets

    plot_latency_time_buckets(all_system_buckets, outfile="latency_time_buckets.png")

if __name__ == "__main__":
    main()