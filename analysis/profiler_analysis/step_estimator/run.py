import argparse
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ----------------------------
# Helper functions
# ----------------------------

def compute_stats(arr, batch_sizes):
    arr = np.array(arr)
    return {
        "count": len(arr),
        "avg": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "stddev": float(np.std(arr)),
        "total_samples_processed": int(np.sum(batch_sizes)),
    }


def parse_timestamp(ts):
    # Example: "2025-11-19_18:04:24.933"
    return datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S.%f")


# ----------------------------
# Load JSONL entries
# ----------------------------

def load_entries(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


# ----------------------------
# Analyze a single predict profiler log
# ----------------------------

def analyze_file(input_path):
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        return

    entries = load_entries(input_path)

    timestamps = []
    predict_times = []
    batch_sizes = []

    for e in entries:
        timestamps.append(parse_timestamp(e["timestamp"]))
        predict_times.append(e["sections"]["predict"])
        batch_sizes.append(e["info"]["batch_size"])

    # Sort chronologically
    zipped = sorted(zip(timestamps, predict_times, batch_sizes))
    timestamps, predict_times, batch_sizes = map(list, zip(*zipped))

    # Stats
    stats = compute_stats(predict_times, batch_sizes)

    # Output paths
    base_dir = os.path.dirname(os.path.abspath(input_path))

    out_report = os.path.join(base_dir, f"predict_summary.txt")
    out_timeplot = os.path.join(base_dir, f"predict_vs_time.png")
    out_batchplot = os.path.join(base_dir, f"predict_vs_batch.png")

    # ----------------------------
    # Write summary
    # ----------------------------
    with open(out_report, "w") as f:
        f.write("=== PREDICT LATENCY SUMMARY ===\n\n")
        f.write(f"count   : {stats['count']}\n")
        f.write(f"avg(ms) : {stats['avg']:.4f}\n")
        f.write(f"median  : {stats['median']:.4f}\n")
        f.write(f"p95     : {stats['p95']:.4f}\n")
        f.write(f"min     : {stats['min']:.4f}\n")
        f.write(f"max     : {stats['max']:.4f}\n")
        f.write(f"stddev  : {stats['stddev']:.4f}\n")
        f.write(f"total_samples_processed: {stats['total_samples_processed']}\n")

    # Print to stdout
    print("=== PREDICT LATENCY SUMMARY ===\n")
    print(f"count   : {stats['count']}")
    print(f"total_samples_processed: {stats['total_samples_processed']}")

    # ----------------------------
    # Plot: predict time vs timestamp
    # ----------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, predict_times, marker=".", linewidth=1)
    plt.title(f"Predict Latency vs Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Predict Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_timeplot)
    plt.close()

    # ----------------------------
    # Plot: predict time vs batch size
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(batch_sizes, predict_times, alpha=0.7)
    plt.title(f"Predict Latency vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Predict Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_batchplot)
    plt.close()

    print(f"[done] analyzed {input_path}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze predict() profiler logs.")
    parser.add_argument("path", help="Path to predict-profiler JSONL file")
    args = parser.parse_args()

    analyze_file(args.path)


if __name__ == "__main__":
    main()
