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
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ------------------------------------------------------------
# Main analysis for executor profiler file
# ------------------------------------------------------------

def analyze_executor_file(input_path, output_dir):
    entries = load_entries(input_path)
    if not entries:
        print(f"[WARN] No valid entries in {input_path}")
        return

    fname = os.path.basename(input_path)
    name_no_ext = os.path.splitext(fname)[0]

    # Output file paths
    out_report = os.path.join(output_dir, f"{name_no_ext}_summary.txt")
    out_lineplot = os.path.join(output_dir, f"executor_over_time_{name_no_ext}.png")
    out_scatter = os.path.join(output_dir, f"executor_scatter_{name_no_ext}.png")

    # Data storage
    exec_latencies = []  # execute_model values
    exec_timestamps = []
    tokens_map = defaultdict(list)   # num_tokens → [latencies]
    scatter_x = []
    scatter_y = []

    for e in entries:
        sec = e.get("sections", {})
        info = e.get("info", {})
        ts = e.get("timestamp")

        if "execute_model" not in sec:
            continue

        latency = sec["execute_model"]
        exec_latencies.append(latency)

        # parse timestamp
        try:
            exec_timestamps.append(datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S.%f"))
        except Exception:
            continue

        num_tokens = info.get("num_tokens")
        if num_tokens is not None:
            tokens_map[num_tokens].append(latency)
            scatter_x.append(num_tokens)
            scatter_y.append(latency)

    # -------------------------------------------------
    # Write summary file
    # -------------------------------------------------

    with open(out_report, "w") as f:
        f.write("=== EXECUTOR OVERHEAD SUMMARY ===\n")
        f.write(f"file: {input_path}\n")
        f.write(f"total_entries: {len(exec_latencies)}\n\n")

        if not exec_latencies:
            f.write("[ERROR] No execute_model entries found.\n")
            return

        lat = np.array(exec_latencies)
        avg = lat.mean()

        f.write(f"avg_execute_model = {avg:.4f} ms\n")
        f.write(f"median = {np.median(lat):.4f} ms\n")
        f.write(f"p95 = {p95(lat):.4f} ms\n")
        f.write(f"min = {lat.min():.4f} ms\n")
        f.write(f"max = {lat.max():.4f} ms\n")
        f.write(f"stddev = {lat.std():.4f}\n")

        # -------------------------------------------------
        # TABLE — grouped by num_tokens
        # -------------------------------------------------

        f.write("\n\n=== GROUPED BY num_tokens ===\n")
        f.write(
            f"{'num_tokens':12} | {'count':>6} | {'avg':>8} | {'med':>8} | "
            f"{'p95':>8} | {'min':>8} | {'max':>8} | {'stddev':>8}\n"
        )
        f.write("-" * 90 + "\n")

        for tok in sorted(tokens_map.keys()):
            vals = np.array(tokens_map[tok])
            f.write(
                f"{tok:12d} | "
                f"{len(vals):6d} | "
                f"{fmt(vals.mean()):>8} | "
                f"{fmt(np.median(vals)):>8} | "
                f"{fmt(p95(vals)):>8} | "
                f"{fmt(vals.min()):>8} | "
                f"{fmt(vals.max()):>8} | "
                f"{fmt(vals.std()):>8}\n"
            )

    # -------------------------------------------------
    # Line plot: execute_model over time
    # -------------------------------------------------
    if exec_latencies and exec_timestamps:
        plt.figure(figsize=(12, 4))
        plt.plot(exec_timestamps, exec_latencies, marker=".", linewidth=1)
        plt.title(f"Executor Latency Over Time ({name_no_ext})")
        plt.xlabel("Timestamp")
        plt.ylabel("execute_model (ms)")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_lineplot)
        plt.close()

    # -------------------------------------------------
    # Scatter plot: latency vs num_tokens
    # -------------------------------------------------
    if scatter_x:
        plt.figure(figsize=(8, 6))
        plt.scatter(scatter_x, scatter_y, alpha=0.5)
        plt.xlabel("num_tokens")
        plt.ylabel("execute_model (ms)")
        plt.title(f"Executor Latency vs num_tokens ({name_no_ext})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_scatter)
        plt.close()


# ------------------------------------------------------------
# Directory mode
# ------------------------------------------------------------

def analyze_directory(dir_path):
    files = sorted(os.listdir(dir_path))
    for fname in files:
        if fname.endswith(".jsonl") or fname.endswith(".json"):
            full = os.path.join(dir_path, fname)
            print(f"[INFO] Processing {full}")
            analyze_executor_file(full, dir_path)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze executor model latency profiler logs.")
    parser.add_argument("path", help="Path to executor JSONL file OR directory")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        analyze_directory(args.path)
    else:
        analyze_executor_file(args.path, os.path.dirname(os.path.abspath(args.path)))


if __name__ == "__main__":
    main()
