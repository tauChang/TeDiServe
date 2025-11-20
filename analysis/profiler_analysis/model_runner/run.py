import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def pct(a, b):
    return (a / b * 100.0) if b else 0.0

def p95(v):
    return float(np.percentile(v, 95)) if len(v) else 0.0

def fmt(x):
    return f"{x:.4f}"


# ------------------------------------------------------------
# Load JSONL entries
# ------------------------------------------------------------

def load_entries(path):
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


# ------------------------------------------------------------
# Main analysis for a single file
# ------------------------------------------------------------

def analyze_file(input_path, output_dir):
    entries = load_entries(input_path)
    if not entries:
        print(f"[WARN] No valid entries in {input_path}")
        return

    fname = os.path.basename(input_path)
    name_no_ext = os.path.splitext(fname)[0]

    # Output file paths
    out_report = os.path.join(output_dir, f"{name_no_ext}_summary.txt")
    out_lineplot = os.path.join(output_dir, f"model_overhead_plot_{name_no_ext}.png")
    out_scatter = os.path.join(output_dir, f"model_overhead_scatter_{name_no_ext}.png")

    # Data aggregations
    section_values = defaultdict(list)
    tokens_values = defaultdict(list)
    scatter_x = []  # num_input_tokens
    scatter_y = []  # total latency

    for e in entries:
        sec = e["sections"]
        info = e.get("info", {})
        num_tokens = info.get("num_input_tokens")

        # per-section stats
        for k, v in sec.items():
            section_values[k].append(v)

        # grouped by input tokens
        if num_tokens is not None:
            tokens_values[num_tokens].append(sec["total"])
            scatter_x.append(num_tokens)
            scatter_y.append(sec["total"])

    # ----------------------------------
    # Write TEXT summary
    # ----------------------------------
    with open(out_report, "w") as f:

        # ====================================================
        # TABLE 1 — overall latency breakdown
        # ====================================================
        f.write("=== TABLE 1: Overall Latency Breakdown ===\n")

        if "total" not in section_values:
            f.write("Error: no 'total' field found.\n")
            return

        total_avg = np.mean(section_values["total"])
        f.write(f"total_avg_latency = {total_avg:.4f} ms\n\n")

        f.write(f"{'stage':24} | {'avg(ms)':>10} | {'%total':>8}\n")
        f.write("-" * 50 + "\n")

        for stage in sorted(section_values.keys()):
            avg = np.mean(section_values[stage])
            f.write(
                f"{stage:24} | {avg:10.4f} | {pct(avg, total_avg):8.2f}%\n"
            )

        # ====================================================
        # TABLE 2 — per-stage statistics
        # ====================================================
        f.write("\n\n=== TABLE 2: Per-Stage Statistics (All Entries) ===\n")

        header = (
            f"{'stage':24} | {'count':>6} | {'avg':>8} | {'med':>8} | "
            f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for stage in sorted(section_values.keys()):
            vals = np.array(section_values[stage])

            f.write(
                f"{stage:24} | "
                f"{len(vals):6d} | "
                f"{fmt(vals.mean()):>8} | "
                f"{fmt(np.median(vals)):>8} | "
                f"{fmt(p95(vals)):>8} | "
                f"{fmt(vals.min()):>8} | "
                f"{fmt(vals.max()):>8} | "
                f"{fmt(vals.std()):>8}\n"
            )

        # ====================================================
        # TABLE 3 — grouped by num_input_tokens
        # ====================================================
        f.write("\n\n=== TABLE 3: Per-num_input_tokens Total Latency ===\n")

        f.write(
            f"{'num_tokens':12} | {'count':6} | {'avg':>8} | {'med':>8} | "
            f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}\n"
        )
        f.write("-" * 90 + "\n")

        for num_tok in sorted(tokens_values.keys()):
            vals = np.array(tokens_values[num_tok])
            f.write(
                f"{num_tok:12d} | "
                f"{len(vals):6d} | "
                f"{fmt(vals.mean()):>8} | "
                f"{fmt(np.median(vals)):>8} | "
                f"{fmt(p95(vals)):>8} | "
                f"{fmt(vals.min()):>8} | "
                f"{fmt(vals.max()):>8} | "
                f"{fmt(vals.std()):>8}\n"
            )

    # ----------------------------------
    # PLOTS
    # ----------------------------------

    # Line plot of "total" latency
    totals = []
    times = []
    for e in entries:
        totals.append(e["sections"]["total"])
        t = datetime.strptime(e["timestamp"], "%Y-%m-%d_%H:%M:%S.%f")
        times.append(t)

    plt.figure(figsize=(12, 4))
    plt.plot(times, totals, marker=".", linewidth=1)
    plt.title(f"Model Step Latency Over Time ({name_no_ext})")
    plt.xlabel("Timestamp")
    plt.ylabel("Latency (ms)")
    plt.grid(True, alpha=0.3)

    # Rotate timestamp labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(out_lineplot)
    plt.close()

    # Scatter plot: total vs num_input_tokens
    if scatter_x:
        plt.figure(figsize=(8, 6))
        plt.scatter(scatter_x, scatter_y, alpha=0.5)
        plt.xlabel("num_input_tokens")
        plt.ylabel("total latency (ms)")
        plt.title(f"Total Latency vs Input Tokens ({name_no_ext})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_scatter)
        plt.close()


# ------------------------------------------------------------
# Directory mode
# ------------------------------------------------------------

def analyze_directory(dir_path):
    files = sorted(os.listdir(dir_path))
    out_dir = dir_path  # write outputs in same directory

    for fname in files:
        if fname.endswith(".jsonl") or fname.endswith(".json"):
            full_path = os.path.join(dir_path, fname)
            print(f"[INFO] Processing {full_path}...")
            analyze_file(full_path, out_dir)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze model overhead from profiler logs.")
    parser.add_argument("path", help="Path to profiler JSONL file OR directory of files")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        analyze_directory(args.path)
    else:
        analyze_file(args.path, os.path.dirname(os.path.abspath(args.path)))


if __name__ == "__main__":
    main()
