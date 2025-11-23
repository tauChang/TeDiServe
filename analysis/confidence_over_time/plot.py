import json
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import bisect
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Any, Dict


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def _active_at_time_local(t: datetime,
                          arrival_list: List[datetime],
                          completion_list: List[datetime]) -> int:
    """Local helper for workers: compute #active requests at time t."""
    arrived = bisect.bisect_right(arrival_list, t)
    completed = bisect.bisect_right(completion_list, t)
    return arrived - completed


def _plot_single_request(args: Tuple[Any, List[dict], datetime, datetime, float,
                                      List[datetime], List[datetime], str]) -> str:
    """
    Worker function to plot a single request's figure.

    Args tuple:
        rid,
        entries,
        start,
        end,
        slo_seconds,
        arrival_list,
        completion_list,
        output_dir
    """
    (rid, entries, start, end, slo_seconds,
     arrival_list, completion_list, output_dir) = args

    deadline = start + timedelta(seconds=slo_seconds)

    timestamps = [e["timestamp"] for e in entries]
    conf = [e["confidence_threshold"] for e in entries]
    max_conf = [e["max_confidence_threshold"] for e in entries]

    active_times = []
    active_vals = []
    for ts in timestamps:
        if start <= ts <= end:
            active_times.append(ts)
            active_vals.append(_active_at_time_local(ts, arrival_list, completion_list))

    all_times = active_times + timestamps + [start, end, deadline]
    min_time = min(all_times) - timedelta(seconds=0.5)
    max_time = max(all_times) + timedelta(seconds=0.5)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    fig.suptitle(f"Request {rid}")

    # Upper plot
    axs[0].set_ylim(0.5, 1.0)
    axs[0].plot(timestamps, max_conf, label="max_confidence_threshold", marker="o")
    axs[0].plot(timestamps, conf, label="confidence_threshold", marker="x")
    axs[0].axvline(deadline, color="red", linestyle="--", label="SLO deadline")
    axs[0].set_ylabel("confidence")
    axs[0].grid(True)
    axs[0].legend()

    # Lower plot
    color = axs[1].plot([], [])[0].get_color()
    axs[1].step(active_times, active_vals, where="post", color=color, linewidth=2)
    axs[1].plot(active_times, active_vals, "o", color=color)
    axs[1].axvline(deadline, color="red", linestyle="--")

    axs[1].set_ylabel("# active requests")
    axs[1].set_xlabel("time")
    axs[1].set_xlim(min_time, max_time)
    axs[1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"individual/{rid}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main(step_data_path: str, workload_history_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Load step_data.json
    # -------------------------------------------------------
    step_data: Dict[Any, List[dict]] = defaultdict(list)

    with open(step_data_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry["id"]
            # if rid contains "warmup", skip
            if "warmup" in str(rid).lower():
                continue
            ts = parse_timestamp(entry["timestamp"])
            entry["timestamp"] = ts
            step_data[rid].append(entry)

    for rid in step_data:
        step_data[rid].sort(key=lambda x: x["timestamp"])

    # -------------------------------------------------------
    # Load workload_history.json
    # -------------------------------------------------------
    arrival: Dict[Any, datetime] = {}
    completion: Dict[Any, datetime] = {}
    latency_slo: Dict[Any, float] = {}

    with open(workload_history_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            rid = entry["request_id"]

            if "arrival_time" in entry:
                t = parse_timestamp(entry["arrival_time"])
                arrival[rid] = t
                latency_slo[rid] = entry["latency_slo"]

            if "completion_time" in entry:
                completion[rid] = parse_timestamp(entry["completion_time"])

    arrival_list = sorted(arrival.values())
    completion_list = sorted(completion.values())

    # -------------------------------------------------------
    # Per-request plots (parallel)
    # -------------------------------------------------------
    tasks = []
    for rid, entries in step_data.items():
        if rid not in arrival or rid not in completion:
            continue

        tasks.append(
            (
                rid,
                entries,
                arrival[rid],
                completion[rid],
                latency_slo[rid],
                arrival_list,
                completion_list,
                output_dir,
            )
        )

    # if tasks:
    #     with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
    #         for out_path in pool.map(_plot_single_request, tasks):
    #             print(f"Saved {out_path}")
    
        # -------------------------------------------------------
    # NEW: PER-JOB + OVERALL CONFIDENCE + STEPS STATS
    # -------------------------------------------------------

    per_job_vals = defaultdict(list)
    per_job_steps = defaultdict(int)

    for rid, entries in step_data.items():
        for e in entries:
            val = e.get("confidence_threshold")
            if val is not None:
                per_job_vals[rid].append(val)
            per_job_steps[rid] += 1

    # Per-job averages
    per_job_avg_conf = {
        rid: (sum(v) / len(v)) for rid, v in per_job_vals.items() if v
    }

    # Flatten all confidence values
    all_vals = [v for lst in per_job_vals.values() for v in lst]
    overall_avg_conf = sum(all_vals) / len(all_vals) if all_vals else None

    # Aggregate step statistics
    total_steps = sum(per_job_steps.values())
    num_jobs = len(per_job_steps)
    overall_avg_steps_per_job = total_steps / num_jobs if num_jobs else None

    # Build structured output
    per_job_output = {}
    for rid in per_job_steps:
        per_job_output[rid] = {
            "num_steps": per_job_steps[rid],
            "avg_confidence_threshold": per_job_avg_conf.get(rid, None),
        }

    out_json = {
        "aggregate": {
            "overall_avg_confidence_threshold": overall_avg_conf,
            "overall_num_steps": total_steps,
            "overall_avg_steps_per_job": overall_avg_steps_per_job,
            "num_jobs": num_jobs,
            "num_records": len(all_vals),
        },
        "per_job": per_job_output,
    }

    output_path = os.path.join(output_dir, "avg_confidence_stats.json")
    with open(output_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"Saved aggregated confidence stats: {output_path}")


    # -------------------------------------------------------
    # GLOBAL CONFIDENCE OVER TIME
    # -------------------------------------------------------
    req_traces = {}
    for rid, entries in step_data.items():
        req_traces[rid] = [
            (e["timestamp"], e["confidence_threshold"], e["max_confidence_threshold"])
            for e in entries
        ]

    global_times = sorted({ts for rid in req_traces for (ts, _, _) in req_traces[rid]})

    latest_conf = {rid: None for rid in req_traces}
    latest_max_conf = {rid: None for rid in req_traces}
    ptr = {rid: 0 for rid in req_traces}

    global_avg_conf = []
    global_avg_max_conf = []
    global_active_vals = []

    for t in global_times:
        active_confs = []
        active_max_confs = []

        for rid in req_traces:
            if rid not in arrival or rid not in completion:
                continue
            if not (arrival[rid] <= t <= completion[rid]):
                continue

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

        global_active_vals.append(
            _active_at_time_local(t, arrival_list, completion_list)
        )

    # -------------------------------------------------------
    # GLOBAL PLOT
    # -------------------------------------------------------
    if global_times:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
        fig.suptitle("Global Confidence & Active Requests Over Time")

        axs[0].set_ylim(0.5, 1.0)
        axs[0].plot(global_times, global_avg_conf, marker="o", label="avg_confidence")
        axs[0].plot(global_times, global_avg_max_conf, marker="x",
                    label="avg_max_confidence")
        axs[0].set_ylabel("confidence")
        axs[0].grid(True)
        axs[0].legend()

        color = axs[1].plot([], [])[0].get_color()
        axs[1].step(global_times, global_active_vals, where="post",
                    color=color, linewidth=2)
        axs[1].plot(global_times, global_active_vals, "o", color=color)

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
