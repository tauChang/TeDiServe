import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.dates as mdates
from datetime import datetime


# ----------------------------
# Helpers
# ----------------------------

def pct(a, b):
    return (a * 100.0 / b) if b else 0.0

def p95(values):
    """Safe percentile function for lists or NumPy arrays."""
    if values is None:
        return 0.0
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, 95))

def fmt(x):
    return f"{x:.4f}"


# ----------------------------
# Load log entries
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
# Main analysis & plotting
# ----------------------------

def analyze_and_plot(input_path):
    entries = load_entries(input_path)

    # Output paths
    base_dir = os.path.dirname(os.path.abspath(input_path))
    out_report = os.path.join(base_dir, "update_summary.txt")
    out_lineplot = os.path.join(base_dir, "update_plot.png")

    # Aggregations
    section_values = defaultdict(list)
    numreq_values = defaultdict(list)

    for e in entries:
        sec = e["sections"]
        info = e.get("info", {})
        nreq = info.get("num_requests", None)

        for k, v in sec.items():
            section_values[k].append(v)

        if nreq is not None:
            numreq_values[nreq].append(sec["total"])

    # ----------------------------------------
    # Write text summary file
    # ----------------------------------------
    with open(out_report, "w") as f:

        # =====================================================
        # TABLE 1: Overall Latency Breakdown
        # =====================================================
        f.write("=== TABLE 1: Overall Latency Breakdown ===\n")

        if "total" not in section_values:
            f.write("Error: 'total' not found in profiler logs.\n")
            return

        total_avg = np.mean(section_values["total"])
        f.write(f"total_avg_latency = {total_avg:.4f} ms\n\n")

        f.write(f"{'stage':18} | {'avg(ms)':>10} | {'%total':>8}\n")
        f.write("-" * 45 + "\n")

        for stage in sorted(section_values.keys()):
            avg = np.mean(section_values[stage])
            f.write(
                f"{stage:18} | {avg:10.4f} | {pct(avg, total_avg):8.2f}%\n"
            )

        # =====================================================
        # TABLE 2: Per-Stage Statistics
        # =====================================================
        f.write("\n\n=== TABLE 2: Per-Stage Statistics (All Entries) ===\n")
        header = (
            f"{'stage':18} | {'count':>6} | {'avg':>8} | {'med':>8} | "
            f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for stage in sorted(section_values.keys()):
            vals = np.array(section_values[stage])

            f.write(
                f"{stage:18} | "
                f"{len(vals):6d} | "
                f"{fmt(vals.mean()):>8} | "
                f"{fmt(np.median(vals)):>8} | "
                f"{fmt(p95(vals)):>8} | "
                f"{fmt(vals.min()):>8} | "
                f"{fmt(vals.max()):>8} | "
                f"{fmt(vals.std()):>8}\n"
            )

        # =====================================================
        # TABLE 3: Per-num_requests breakdown
        # =====================================================
        f.write("\n\n=== TABLE 3: Per-num_requests Total Latency ===\n")
        f.write(
            f"{'num_req':8} | {'count':6} | {'avg':>8} | {'med':>8} | "
            f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}\n"
        )
        f.write("-" * 90 + "\n")

        for nreq in sorted(numreq_values.keys()):
            vals = np.array(numreq_values[nreq])
            f.write(
                f"{nreq:8d} | "
                f"{len(vals):6d} | "
                f"{fmt(vals.mean()):>8} | "
                f"{fmt(np.median(vals)):>8} | "
                f"{fmt(p95(vals)):>8} | "
                f"{fmt(vals.min()):>8} | "
                f"{fmt(vals.max()):>8} | "
                f"{fmt(vals.std()):>8}\n"
            )

        # ----------------------------------------
    # Line plot: total latency vs timestamp
    # ----------------------------------------

    timestamps = []
    totals = []

    # Load timestamps & totals in the same order as section_values
    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            ts = record["timestamp"]
            total = record["sections"]["total"]

            # Parse timestamp
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S.%f")
            except ValueError:
                dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S")

            timestamps.append(dt)
            totals.append(total)

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, totals, marker=".", linewidth=1)

    plt.title("Total Scheduler Loop Latency Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Latency (ms)")
    
    # Format timestamps nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.gcf().autofmt_xdate()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_lineplot)
    plt.close()

# ----------------------------
# Command-line entrypoint
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze scheduler overhead from profiler JSONL.")
    parser.add_argument("input", help="Path to profiler JSONL file")
    args = parser.parse_args()
    analyze_and_plot(args.input)

if __name__ == "__main__":
    main()
