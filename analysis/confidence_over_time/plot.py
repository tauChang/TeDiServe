import json
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import bisect


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def main(step_data_path: str, workload_history_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Load step_data.json
    # -------------------------------------------------------
    step_data = defaultdict(list)

    with open(step_data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry["id"]
            ts = parse_timestamp(entry["timestamp"])
            entry["timestamp"] = ts
            step_data[rid].append(entry)

    # Sort per-request step timelines
    for rid in step_data:
        step_data[rid].sort(key=lambda x: x["timestamp"])

    # -------------------------------------------------------
    # Load workload_history.json
    # -------------------------------------------------------
    arrival = {}
    completion = {}
    latency_slo = {}

    with open(workload_history_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry["request_id"]

            if "arrival_time" in entry:
                t = parse_timestamp(entry["arrival_time"])
                arrival[rid] = t
                latency_slo[rid] = entry["latency_slo"]

            if "completion_time" in entry:
                t = parse_timestamp(entry["completion_time"])
                completion[rid] = t

    # Precompute sorted lists for binary-search active request lookup
    arrival_list = sorted(arrival.values())
    completion_list = sorted(completion.values())

    # Correct active-at-time function
    def active_at_time(t: datetime):
        arrived = bisect.bisect_right(arrival_list, t)
        completed = bisect.bisect_right(completion_list, t)
        return arrived - completed

    # -------------------------------------------------------
    # Per-request plots
    # -------------------------------------------------------
    for rid in step_data:
        if rid not in arrival or rid not in completion:
            continue

        start = arrival[rid]
        end = completion[rid]
        deadline = start + timedelta(seconds=latency_slo[rid])

        timestamps = [e["timestamp"] for e in step_data[rid]]
        conf = [e["confidence_threshold"] for e in step_data[rid]]
        max_conf = [e["max_confidence_threshold"] for e in step_data[rid]]

        # Compute active request counts at step timestamps
        active_times = []
        active_vals = []
        for ts in timestamps:
            if start <= ts <= end:
                active_times.append(ts)
                active_vals.append(active_at_time(ts))

        # Compute plotting window
        all_times = active_times + timestamps + [start, end, deadline]
        min_time = min(all_times) - timedelta(seconds=0.5)
        max_time = max(all_times) + timedelta(seconds=0.5)

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
        fig.suptitle(f"Request {rid}")

        #------------------------------------------
        # Upper subplot: confidence thresholds
        #------------------------------------------
        axs[0].set_ylim(0.5, 1.0)
        axs[0].plot(timestamps, max_conf, label="max_confidence_threshold", marker="o")
        axs[0].plot(timestamps, conf, label="confidence_threshold", marker="x")
        axs[0].axvline(deadline, color="red", linestyle="--", label="SLO deadline")
        axs[0].set_ylabel("confidence")
        axs[0].grid(True)
        axs[0].legend()

        #------------------------------------------
        # Lower subplot: active request count (step)
        #------------------------------------------
        color = axs[1].plot([], [])[0].get_color()  # get consistent color

        axs[1].step(active_times, active_vals, where="post", color=color, linewidth=2)
        axs[1].plot(active_times, active_vals, 'o', color=color)

        axs[1].axvline(deadline, color="red", linestyle="--")
        axs[1].set_ylabel("# active requests")
        axs[1].set_xlabel("time")
        axs[1].set_xlim(min_time, max_time)
        axs[1].grid(True)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{rid}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    # -------------------------------------------------------
    # GLOBAL CONFIDENCE OVER TIME
    # -------------------------------------------------------
    req_traces = {}
    for rid, entries in step_data.items():
        req_traces[rid] = [
            (e["timestamp"], e["confidence_threshold"], e["max_confidence_threshold"])
            for e in entries
        ]

    # Unified global timeline
    global_times = sorted({ts for rid in req_traces for (ts, _, _) in req_traces[rid]})

    latest_conf = {rid: None for rid in req_traces}
    latest_max_conf = {rid: None for rid in req_traces}
    ptr = {rid: 0 for rid in req_traces}

    global_avg_conf = []
    global_avg_max_conf = []
    global_active_vals = []

    # Compute global metrics
    for t in global_times:
        active_confs = []
        active_max_confs = []

        for rid in req_traces:
            if rid not in arrival or rid not in completion:
                continue
            if not (arrival[rid] <= t <= completion[rid]):
                continue

            # Advance pointer
            while ptr[rid] < len(req_traces[rid]) and req_traces[rid][ptr[rid]][0] <= t:
                latest_conf[rid] = req_traces[rid][ptr[rid]][1]
                latest_max_conf[rid] = req_traces[rid][ptr[rid]][2]
                ptr[rid] += 1

            if latest_conf[rid] is not None:
                active_confs.append(latest_conf[rid])
                active_max_confs.append(latest_max_conf[rid])

        if active_confs:
            global_avg_conf.append(sum(active_confs) / len(active_confs))
            global_avg_max_conf.append(sum(active_max_confs) / len(active_max_confs))
        else:
            global_avg_conf.append(None)
            global_avg_max_conf.append(None)

        global_active_vals.append(active_at_time(t))

    # -------------------------------------------------------
    # GLOBAL PLOT (step)
    # -------------------------------------------------------
    if global_times:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
        fig.suptitle("Global Confidence & Active Requests Over Time")

        # Upper: global confidence curves
        axs[0].set_ylim(0.5, 1.0)
        axs[0].plot(global_times, global_avg_conf, marker="o", label="avg_confidence")
        axs[0].plot(global_times, global_avg_max_conf, marker="x", label="avg_max_confidence")
        axs[0].set_ylabel("confidence")
        axs[0].grid(True)
        axs[0].legend()

        # Lower: active requests (correct step plot)
        color = axs[1].plot([], [])[0].get_color()
        axs[1].step(global_times, global_active_vals, where="post", color=color, linewidth=2)
        axs[1].plot(global_times, global_active_vals, 'o', color=color)

        axs[1].set_ylabel("# active requests")
        axs[1].set_xlabel("time")
        axs[1].grid(True)

        plt.tight_layout()
        out_path = os.path.join(output_dir, "global_confidence.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved global plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-data", required=True)
    parser.add_argument("--workload-history", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.step_data, args.workload_history, args.output_dir)
