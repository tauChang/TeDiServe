import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np


def read_system_log(path):
    snapshots, buf = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            buf.append(line)
            if line == "}":
                obj_str = "\n".join(buf)
                try:
                    snapshots.append(json.loads(obj_str))
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed JSON block: {e}")
                buf = []
    return snapshots


def plot_executor_mapping(log_path):
    snapshots = read_system_log(log_path)

    if not snapshots:
        print("No snapshots found.")
        return

    # Parse timestamps
    for s in snapshots:
        s["timestamp"] = datetime.strptime(s["timestamp"], "%Y-%m-%d_%H:%M:%S.%f")

    # ✅ Normalize time to start at 0 seconds
    t0 = min(s["timestamp"] for s in snapshots)
    for s in snapshots:
        s["time_rel"] = (s["timestamp"] - t0).total_seconds()

    # Flatten executor-request mapping
    rows = []
    for s in snapshots:
        ts = s["time_rel"]   # ⬅️ use relative time
        for exec_id, reqs in s["executor_to_requests"].items():
            for r in reqs:
                rows.append({
                    "time_rel": ts,
                    "executor": int(exec_id),
                    "request": r
                })

    df = pd.DataFrame(rows)
    print(df)
    if df.empty:
        print("No executor-request mappings found.")
        return

    df = df.sort_values("time_rel")

    # Get executor TP degrees from last snapshot
    last_tp = snapshots[-1].get("executor_tp_degree", {})
    executor_tp = {int(k): v for k, v in last_tp.items()}

    # Sort executors by TP degree, then ID (ascending)
    exec_order = sorted(executor_tp.keys(), key=lambda k: (executor_tp[k], k))
    exec_labels = [f"Executor {i} (TP={executor_tp.get(i, '?')})" for i in exec_order]

    # Distinct color map for requests
    request_ids = df["request"].unique()
    base_colors = plt.get_cmap("tab10").colors
    color_map = {rid: base_colors[i % len(base_colors)] for i, rid in enumerate(request_ids)}
    df["color"] = df["request"].map(color_map)

    # Plot
    plt.figure(figsize=(10, 5))
    for exec_id, sub in df.groupby("executor"):
        plt.scatter(
            sub["time_rel"],
            [exec_order.index(exec_id)] * len(sub),
            c=sub["color"],
            s=60,
            marker="s",
        )

    # Discrete y-axis with labels ordered by TP degree
    plt.yticks(range(len(exec_order)), exec_labels)

    # Build legend for requests
    patches = [mpatches.Patch(color=color_map[rid], label=rid) for rid in request_ids]
    plt.legend(handles=patches, title="Requests", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xlabel("Time (s since start)", fontsize=12)
    plt.ylabel("Executor (sorted by TP degree → ID)", fontsize=12)
    plt.title("Executor → Request Mapping Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig("no opp.png", dpi=300)
    plt.close()
    print("Saved: executor_request_mapping.png")



# Example usage
plot_executor_mapping("no opp.log")
