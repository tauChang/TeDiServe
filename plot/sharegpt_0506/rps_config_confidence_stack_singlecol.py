#!/usr/bin/env python3
import argparse
import bisect
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = BASE_DIR / "sharegpt_0506_timeline_summary.json"
ARRIVAL_SNAPSHOT = BASE_DIR / "sharegpt_0506_arrivals.txt"
OUTPUT_PDF = BASE_DIR / "full_timeline_singlecol.pdf"
OUTPUT_PNG = BASE_DIR / "full_timeline_singlecol.png"

SOURCE_CONFIG = {
    "tedi_base": Path(
        "/u/tchang85/dllm/sbatch_experiment_dir/20260506/"
        "114447_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/"
    ),
    "arrival_file": Path("/u/tchang85/dllm/BurstGPT/burstgpt_2hrs_20qps.txt"),
    "slo_seconds": 12.5,
    "bucket_seconds": 180,
    "slo_baselines": {
        "TeDiServe": Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260506/"
            "114447_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/logs/"
            "benchmark_latency_info.json"
        ),
        "Llumnix": Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260503/"
            "163209_sharegpt_20_llumnix_only/0_llumnix_20qps/"
        ),
        "INFaaS": Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260503/"
            "193754_sharegpt_20_infaas_only/0_infaas_20qps/"
        ),
    },
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
    }
)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")


def dt_to_ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def bucket_ns(t_ns: int, resolution_s: int) -> int:
    res = int(resolution_s * 1e9)
    return (t_ns // res) * res


def resolve_benchmark_latency_info(path: Path) -> Path:
    if path.name == "benchmark_latency_info.json":
        return path
    return path / "logs" / "benchmark_latency_info.json"


def load_arrivals(arrival_file: Path) -> np.ndarray:
    with arrival_file.open() as f:
        vals = [float(x.strip()) for x in f if x.strip()]
    return np.array(vals)


def compute_qps_from_arrivals(arrivals: np.ndarray, window_minutes: float):
    arrivals = arrivals - arrivals.min()
    window_sec = window_minutes * 60

    total_seconds = arrivals.max()
    num_windows = int(np.ceil(total_seconds / window_sec))
    edges = np.arange(0, (num_windows + 1) * window_sec, window_sec)
    counts, _ = np.histogram(arrivals, bins=edges)
    qps = counts / window_sec
    qps_time = np.arange(len(qps)) * window_minutes
    return qps_time, qps


def load_reconfig_trace(json_file: Path):
    trace = json.load(json_file.open())
    trace = trace[:-1] if len(trace) > 1 else trace

    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in trace]
    t0 = timestamps[0]
    rel_times = [(t - t0).total_seconds() / 60.0 for t in timestamps]

    tp_counts = {1: [], 2: [], 4: []}
    for entry in trace:
        counts = {1: 0, 2: 0, 4: 0}
        for workers in entry["config"].values():
            tp = len(workers)
            if tp in counts:
                counts[tp] += 1
        for tp in tp_counts:
            tp_counts[tp].append(counts[tp])

    data = {"time": rel_times}
    for tp, values in tp_counts.items():
        if any(value > 0 for value in values):
            data[f"TP-{tp}"] = values
    return data


def build_raw_cache(step_data_path: Path) -> Path:
    cache_path = step_data_path.with_suffix(step_data_path.suffix + ".raw_cache.npz")
    if cache_path.exists():
        print(f"Using existing confidence cache: {cache_path}")
        return cache_path

    print("Building confidence cache (first time only)...")
    ts_list, conf_list, max_list = [], [], []

    with step_data_path.open() as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("id")
            if "warmup" in str(rid).lower():
                continue
            conf = row.get("confidence_threshold")
            max_conf = row.get("max_confidence_threshold")
            if conf is None or max_conf is None:
                continue
            ts_list.append(dt_to_ns(parse_ts(row["timestamp"])))
            conf_list.append(conf)
            max_list.append(max_conf)

    np.savez(
        cache_path,
        timestamps_ns=np.array(ts_list, np.int64),
        conf=np.array(conf_list, np.float32),
        max_conf=np.array(max_list, np.float32),
    )
    print(f"Cache saved: {cache_path}")
    return cache_path


def build_active_events(workload_file: Path):
    arrival, completion = {}, {}
    with workload_file.open() as f:
        for line in f:
            row = json.loads(line)
            rid = row["request_id"]
            if "arrival_time" in row:
                arrival[rid] = dt_to_ns(parse_ts(row["arrival_time"]))
            if "completion_time" in row:
                completion[rid] = dt_to_ns(parse_ts(row["completion_time"]))

    events = []
    for rid in arrival:
        events.append((arrival[rid], 1))
    for rid in completion:
        events.append((completion[rid], -1))

    events.sort()
    times, deltas = zip(*events)
    prefix = []
    current = 0
    for delta in deltas:
        current += delta
        prefix.append(current)
    return np.array(times), np.array(prefix)


def active_at(t_ns: int, times_ns: np.ndarray, prefix: np.ndarray) -> int:
    idx = bisect.bisect_right(times_ns, t_ns) - 1
    return int(prefix[idx]) if idx >= 0 else 0


def compute_confidence_timeline(cache_path: Path, workload_file: Path, bucket_seconds: int):
    cache = np.load(cache_path)
    ts = cache["timestamps_ns"]
    conf = cache["conf"]
    max_conf = cache["max_conf"]

    bucketed = np.array([bucket_ns(t, bucket_seconds) for t in ts], np.int64)
    buckets, inv = np.unique(bucketed, return_inverse=True)

    avg_conf = np.bincount(inv, weights=conf) / np.bincount(inv)
    avg_max_conf = np.bincount(inv, weights=max_conf) / np.bincount(inv)

    times_ns, prefix = build_active_events(workload_file)
    active_vals = [active_at(t, times_ns, prefix) for t in buckets]

    t0 = buckets[0]
    times_min = (buckets - t0) / 1e9 / 60.0
    return times_min, avg_conf, avg_max_conf, active_vals


def compute_slo_attainment_timeline(benchmark_latency_info_path: Path, slo_seconds: float, bucket_seconds: int):
    benchmark_latency_info_path = resolve_benchmark_latency_info(benchmark_latency_info_path)

    with benchmark_latency_info_path.open() as f:
        data = json.load(f)

    root = data[0] if isinstance(data, list) else data
    instances = root["instances"] if "instances" in root else root

    records = []
    for req in instances.values():
        if not isinstance(req, dict):
            continue
        arrival_time = req.get("arrival_time")
        request_latency = req.get("request_latency")
        status = req.get("status")
        if arrival_time is None or request_latency is None or status is None:
            continue
        records.append(
            (
                dt_to_ns(parse_ts(arrival_time)),
                status == "success" and float(request_latency) < slo_seconds,
            )
        )

    if not records:
        return np.array([]), np.array([])

    records.sort(key=lambda item: item[0])
    timestamps_ns = np.array([record[0] for record in records], dtype=np.int64)
    attained = np.array([record[1] for record in records], dtype=np.int64)

    bucketed = np.array([bucket_ns(t, bucket_seconds) for t in timestamps_ns], dtype=np.int64)
    buckets, inv = np.unique(bucketed, return_inverse=True)

    total_count = np.bincount(inv)
    attained_count = np.bincount(inv, weights=attained)
    attainment = attained_count / total_count * 100.0

    t0 = buckets[0]
    times_min = (buckets - t0) / 1e9 / 60.0
    return times_min, attainment


def to_float_list(values) -> list:
    return [float(value) for value in values]


def to_int_list(values) -> list:
    return [int(value) for value in values]


def build_summary(refresh: bool = False) -> dict:
    tedi_base = SOURCE_CONFIG["tedi_base"]
    arrival_source = SOURCE_CONFIG["arrival_file"]
    bucket_seconds = SOURCE_CONFIG["bucket_seconds"]
    slo_seconds = SOURCE_CONFIG["slo_seconds"]

    if refresh or not ARRIVAL_SNAPSHOT.exists():
        shutil.copy2(arrival_source, ARRIVAL_SNAPSHOT)
        print(f"Saved arrival snapshot to {ARRIVAL_SNAPSHOT}")

    config_json = tedi_base / "config_history.json"
    step_data_file = tedi_base / "step_data.json"
    workload_history_file = tedi_base / "workload_history.json"

    cfg_data = load_reconfig_trace(config_json)
    qps_time, qps_vals = compute_qps_from_arrivals(load_arrivals(ARRIVAL_SNAPSHOT), 1.0)
    cache_path = build_raw_cache(step_data_file)
    conf_time, avg_conf, avg_max_conf, active_vals = compute_confidence_timeline(
        cache_path,
        workload_history_file,
        bucket_seconds=bucket_seconds,
    )

    slo_timelines = {}
    for name, path in SOURCE_CONFIG["slo_baselines"].items():
        times_min, attainment = compute_slo_attainment_timeline(
            path,
            slo_seconds=slo_seconds,
            bucket_seconds=bucket_seconds,
        )
        slo_timelines[name] = {
            "time": to_float_list(times_min),
            "attainment": to_float_list(attainment),
        }

    summary = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "bucket_seconds": bucket_seconds,
            "slo_seconds": slo_seconds,
            "arrival_snapshot": str(ARRIVAL_SNAPSHOT),
            "source_paths": {
                "tedi_base": str(tedi_base),
                "arrival_file": str(arrival_source),
                "config_history": str(config_json),
                "step_data": str(step_data_file),
                "workload_history": str(workload_history_file),
            },
        },
        "qps": {
            "time": to_float_list(qps_time),
            "value": to_float_list(qps_vals),
        },
        "config": {
            key: to_int_list(value) if key != "time" else to_float_list(value)
            for key, value in cfg_data.items()
        },
        "confidence": {
            "time": to_float_list(conf_time),
            "avg_conf": to_float_list(avg_conf),
            "avg_max_conf": to_float_list(avg_max_conf),
            "active_requests": to_int_list(active_vals),
        },
        "slo": slo_timelines,
    }

    with SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary cache to {SUMMARY_JSON}")
    return summary


def load_or_build_summary(refresh: bool = False) -> dict:
    if refresh or not SUMMARY_JSON.exists():
        return build_summary(refresh=refresh)

    with SUMMARY_JSON.open() as f:
        summary = json.load(f)

    # Backward-compatibility for cached summaries created before the INFaaS
    # label rename.
    slo_summary = summary.get("slo", {})
    if "INFaaS" not in slo_summary and "InFaaS" in slo_summary:
        slo_summary["INFaaS"] = slo_summary.pop("InFaaS")

    print(f"Loaded cached summary from {SUMMARY_JSON}")
    return summary


def plot_single_column(summary: dict):
    fig, axs = plt.subplots(
        4,
        1,
        figsize=(3.45, 2.95),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1]},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)

    qps_time = np.array(summary["qps"]["time"])
    qps_vals = np.array(summary["qps"]["value"])
    cfg_data = summary["config"]
    conf_data = summary["confidence"]
    slo_timelines = summary["slo"]

    axs[0].plot(qps_time, qps_vals, color="black")
    axs[0].set_ylabel("RPS")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].set_yticks([0, 10, 20])

    cfg_time = np.array(cfg_data["time"])
    dt = cfg_time[-1] - cfg_time[-2] if len(cfg_time) > 1 else 1.0
    edges = np.append(cfg_time, cfg_time[-1] + dt)
    tp_colors = {"TP-1": "#FFCF71", "TP-2": "#B6771D", "TP-4": "#7B542F"}
    bottom = np.zeros(len(cfg_time))

    for tp in ["TP-4", "TP-2", "TP-1"]:
        if tp not in cfg_data:
            continue
        y = np.array(cfg_data[tp])
        y_top = bottom + y
        axs[1].fill_between(
            edges,
            np.append(bottom, bottom[-1]),
            np.append(y_top, y_top[-1]),
            step="post",
            color=tp_colors[tp],
            alpha=0.85,
            label=tp,
            edgecolor=None,
        )
        bottom = y_top

    axs[1].set_ylabel("# Instances")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].set_ylim(0, max(1, bottom.max() * 1.1))
    if bottom.max() > 0:
        axs[1].set_yticks(np.arange(0, bottom.max() * 1.1 + 0.01, 5))
    handles, labels = axs[1].get_legend_handles_labels()
    order = ["TP-1", "TP-2", "TP-4"]
    present_order = [tp for tp in order if tp in labels]
    if present_order:
        axs[1].legend(
            [handles[labels.index(tp)] for tp in present_order],
            present_order,
            # labelspacing=0.1,
            # borderpad=0.2,
            # handletextpad=0.3,
            # handlelength=1.0,
            labelspacing=0.2,
            handlelength=1.0,
            framealpha=0.8,
            loc="lower right",
        )

    conf_time = np.array(conf_data["time"])
    avg_conf = np.array(conf_data["avg_conf"])
    avg_max_conf = np.array(conf_data["avg_max_conf"])
    axs[2].plot(conf_time, avg_conf, color="#4BB4AA", label="Used")
    axs[2].plot(conf_time, avg_max_conf, color="#540863", label="Max")
    axs[2].set_ylabel("Conf.")
    axs[2].grid(True, linestyle="--", alpha=0.5)
    axs[2].legend(
        [axs[2].lines[1], axs[2].lines[0]],
        ["Max Confidence Threshold", "Confidence Threshold Used"],
        # labelspacing=0.1,
        # borderpad=0.2,
        # handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
        handlelength=1.0,
        borderpad=0.25,
        loc="lower right",
    )
    # move legend slightly up to avoid overlap with x-axis labels
    leg = axs[2].get_legend()
    leg.set_bbox_to_anchor((1, -0.05))
    axs[2].set_ylim(0.6, 0.92)
    axs[2].set_xlim(0, 120)
    axs[2].set_yticks([0.6, 0.7, 0.8, 0.9])

    slo_colors = {"INFaaS": "#134686", "Llumnix": "#B89000", "TeDiServe": "#FF4F0F"}
    slo_order = ["INFaaS", "Llumnix", "TeDiServe"]
    for name in slo_order:
        if name not in slo_timelines:
            continue
        times_min = np.array(slo_timelines[name]["time"])
        attainment = np.array(slo_timelines[name]["attainment"])
        axs[3].plot(times_min, attainment, label=name, color=slo_colors.get(name))

    axs[3].set_ylabel("SLO (%)")
    axs[3].set_xlabel("Time (min)", labelpad=1)
    axs[3].grid(True, linestyle="--", alpha=0.5)
    handles, labels = axs[3].get_legend_handles_labels()
    present_order = [name for name in slo_order if name in labels]
    if present_order:
        axs[3].legend(
            [handles[labels.index(name)] for name in present_order],
            present_order,
            labelspacing=0.25,
            borderpad=0.2,
            handlelength=1.0,
            handletextpad=0.3,
            loc="lower left",
        )
    axs[3].set_ylim(0, 110)
    axs[3].set_xlim(0, 120)
    axs[3].set_yticks([0, 25, 50, 75, 100])
    leg = axs[3].get_legend()
    leg.set_bbox_to_anchor((0, -0.05))

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-summary",
        action="store_true",
        help="Rebuild the local JSON summary and refresh the copied arrival trace.",
    )
    args = parser.parse_args()

    summary = load_or_build_summary(refresh=args.refresh_summary)
    plot_single_column(summary)


if __name__ == "__main__":
    main()