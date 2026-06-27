import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import matplotlib.dates as mdates

def compute_stats(arr):
    arr = np.array(arr)
    return {
        "count": len(arr),
        "avg": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "stddev": float(np.std(arr)),
    }

def analyze_and_plot(input_path):

    base_dir = os.path.dirname(os.path.abspath(input_path))
    out_report = os.path.join(base_dir, "schedule_summary.txt")
    out_lineplot = os.path.join(base_dir, "schedule_plot.png")
    out_boxplot = os.path.join(base_dir, "schedule_boxplot.png")

    buckets = defaultdict(list)

    # ---- Read JSONL ----
    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            num_req = data["info"]["num_requests"]
            sections = data["sections"]
            buckets[num_req].append(sections)

    # ===============================
    # Table 1: Overall latency breakdown
    # ===============================
    stage_values = defaultdict(list)

    for entries in buckets.values():
        for sec in entries:
            for stage, val in sec.items():
                stage_values[stage].append(val)

    stage_avg = {stage: float(np.mean(vals)) for stage, vals in stage_values.items()}
    total_avg = stage_avg.get("total", sum(v for k, v in stage_avg.items() if k != "total"))

    # ===============================
    # Table 2: Per-stage stats
    # ===============================
    per_stage_stats = {stage: compute_stats(vals) for stage, vals in stage_values.items()}

    # ===============================
    # Table 3: Per-num_requests stats
    # ===============================
    per_req_stats = {}
    for num_req, entries in buckets.items():
        totals = [sec["total"] for sec in entries]
        per_req_stats[num_req] = compute_stats(totals)

    # ===============================
    # Write summary text file
    # ===============================
    with open(out_report, "w") as f:

        f.write("=== TABLE 1: Overall Latency Breakdown ===\n")
        f.write(f"total_avg_latency = {total_avg:.4f} ms\n\n")
        f.write(f"{'stage':>10} | {'avg(ms)':>10} | {'%total':>8}\n")
        f.write("-" * 36 + "\n")

        print(f"total_avg_latency = {total_avg:.4f} ms\n\n")

        stage_order = sorted(stage_avg.keys(), key=lambda x: (x != "total", x))

        for stage in stage_order:
            avg = stage_avg[stage]
            pct = (avg / total_avg * 100) if stage != "total" else 100.0
            f.write(f"{stage:10s} | {avg:10.4f} | {pct:7.2f}%\n")

        f.write("\n\n=== TABLE 2: Per-Stage Statistics (All Requests) ===\n")
        f.write(f"{'stage':>10} | {'count':>6} | {'avg':>8} | {'med':>8} | "
                f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}\n")
        f.write("-" * 90 + "\n")

        for stage in stage_order:
            s = per_stage_stats[stage]
            f.write(f"{stage:10s} | {s['count']:6d} | {s['avg']:8.4f} | "
                    f"{s['median']:8.4f} | {s['p95']:8.4f} | {s['min']:8.4f} | "
                    f"{s['max']:8.4f} | {s['stddev']:8.4f}\n")

        f.write("\n\n=== TABLE 3: Per-num_requests Total Latency ===\n")
        f.write(f"{'num_req':>8} | {'count':>6} | {'avg':>8} | {'med':>8} | "
                f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}\n")
        f.write("-" * 90 + "\n")

        for num_req in sorted(per_req_stats.keys()):
            s = per_req_stats[num_req]
            f.write(f"{num_req:8d} | {s['count']:6d} | {s['avg']:8.4f} | "
                    f"{s['median']:8.4f} | {s['p95']:8.4f} | {s['min']:8.4f} | "
                    f"{s['max']:8.4f} | {s['stddev']:8.4f}\n")


    # ===============================
    # Plot 1: Mean latency vs num_requests
    # ===============================
    xs = sorted(per_req_stats.keys())
    ys = [per_req_stats[n]["avg"] for n in xs]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, "-o", linewidth=2)
    plt.xlabel("Number of Requests")
    plt.ylabel("Average Total Latency (ms)")
    plt.title("Scheduler Overhead (Mean Total Latency)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_lineplot, dpi=220)
    plt.close()

    # ===============================
    # Plot 2: Boxplot total latency per bucket
    # ===============================
    data = [[sec["total"] for sec in buckets[n]] for n in xs]

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=xs, showmeans=True)
    plt.xlabel("Number of Requests")
    plt.ylabel("Total Latency (ms)")
    plt.title("Scheduler Overhead Distribution")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_boxplot, dpi=220)
    plt.close()

    # ===============================
    # Plot 3: Latency vs Timestamp
    # ===============================
    timestamps = []
    total_vals = []

    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ts = data["timestamp"]
            total = data["sections"]["total"]

            try:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S")

            timestamps.append(dt)
            total_vals.append(total)

    out_timeplot = os.path.join(base_dir, "schedule_latency_vs_time.png")

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, total_vals, marker=".", linewidth=1)
    plt.title("Scheduler Total Latency Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Total Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_timeplot, dpi=220)
    plt.close()


    # ===============================
    # NEW Plot 4: num_requests over time
    # ===============================
    req_ts = []
    req_counts = []

    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            ts = data["timestamp"]
            nreq = data["info"]["num_requests"]

            try:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S")

            req_ts.append(dt)
            req_counts.append(nreq)

    out_reqplot = os.path.join(base_dir, "schedule_requests_over_time.png")

    plt.figure(figsize=(12, 4))
    plt.plot(req_ts, req_counts, marker="o", linewidth=1.5)
    plt.title("Number of Requests Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("num_requests")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_reqplot, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze scheduler overhead from profiler JSONL.")
    parser.add_argument("input", help="Path to profiler JSONL file")
    args = parser.parse_args()
    analyze_and_plot(args.input)

if __name__ == "__main__":
    main()
