import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_ROOT = Path("/u/tchang85/dllm/reconfig_simulation/reconfig_simulation_outputs")
DEFAULT_TRACE_DIR = Path("/u/tchang85/dllm/BurstGPT")
DEFAULT_OUTPUT = Path("/u/tchang85/dllm/plot/reconfig_ablation/reconfig_ablation.png")
DEFAULT_RPS_VALUES = [2, 4, 6, 8, 10]
PANEL_LAYOUT = [None, 2, 4, 6, 8, 10]
TP_ORDER = ["TP-4", "TP-2", "TP-1"]
TP_COLORS = {"TP-1": "#FFCF71", "TP-2": "#B6771D", "TP-4": "#7B542F"}
ARRIVAL_RPS = 10
ARRIVAL_WINDOW_MIN = 5


def resolve_config_history(input_root: Path, rps: int) -> Path:
    matches = sorted(input_root.glob(f"*_{rps}qps/config_history.json"))
    if not matches:
        raise FileNotFoundError(
            f"No config_history.json found for {rps} QPS under {input_root}"
        )
    return matches[-1]


def load_reconfig_trace(json_file: Path):
    trace = json.load(open(json_file))
    trace = trace[:-1] if len(trace) > 1 else trace

    times_min = [event["trigger_time_s"] / 60.0 for event in trace]
    tp_counts = {1: [], 2: [], 4: []}

    for event in trace:
        counts = {1: 0, 2: 0, 4: 0}
        for workers in event["config"].values():
            tp = len(workers)
            if tp in counts:
                counts[tp] += 1
        for tp in tp_counts:
            tp_counts[tp].append(counts[tp])

    series = {"time": np.array(times_min, dtype=float)}
    for tp, counts in tp_counts.items():
        label = f"TP-{tp}"
        if any(count > 0 for count in counts):
            series[label] = np.array(counts, dtype=float)
    return series


def load_arrivals(arrival_file: Path):
    with open(arrival_file, "r") as handle:
        values = [float(line.strip()) for line in handle if line.strip()]
    arrivals = np.array(values, dtype=float)
    return arrivals - arrivals.min()


def compute_qps_from_arrivals(arrivals, window_minutes: float):
    window_sec = window_minutes * 60.0
    total_time = arrivals.max()
    num_windows = max(1, int(np.ceil(total_time / window_sec)))
    edges = np.arange(0.0, (num_windows + 1) * window_sec, window_sec)
    counts, _ = np.histogram(arrivals, bins=edges)
    qps = counts / window_sec
    times_min = np.arange(len(qps)) * window_minutes
    return times_min, qps


def draw_arrival_axis(ax, trace_dir: Path, y_max: float):
    arrivals = load_arrivals(trace_dir / f"burstgpt_2hrs_{ARRIVAL_RPS}qps.txt")
    times_min, qps = compute_qps_from_arrivals(arrivals, ARRIVAL_WINDOW_MIN)
    ax.plot(times_min, qps, color="black")
    ax.set_title("Arrival Rate", pad=2)
    ax.set_xlim(0, times_min[-1] + ARRIVAL_WINDOW_MIN)
    ax.set_ylim(0, y_max)
    if 60 <= times_min[-1] + ARRIVAL_WINDOW_MIN:
        ax.set_xticks([60])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", direction="in", length=3)
        ax.axvline(60, linestyle="--", alpha=0.5, color="#bdbdbd", zorder=0)
    else:
        ax.set_xticks([])
    ax.set_yticks([])


def draw_config_axis(ax, series, rps: int, y_max: float):
    cfg_time = series["time"]
    if len(cfg_time) == 0:
        ax.set_title(f"RPS {rps}", pad=2)
        return

    dt = cfg_time[-1] - cfg_time[-2] if len(cfg_time) > 1 else 1.0
    edges = np.append(cfg_time, cfg_time[-1] + dt)
    bottom = np.zeros(len(cfg_time), dtype=float)

    for tp in TP_ORDER:
        if tp not in series:
            continue
        y = series[tp]
        y_top = bottom + y
        ax.fill_between(
            edges,
            np.append(bottom, bottom[-1]),
            np.append(y_top, y_top[-1]),
            step="post",
            color=TP_COLORS[tp],
            alpha=0.85,
            edgecolor=None,
            label=tp,
        )
        bottom = y_top

    ax.set_title(f"Max RPS {rps}", pad=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, y_max)
    ax.set_yticks([tick for tick in [0, 8, 16] if tick <= y_max])
    ax.set_xlim(0, edges[-1])
    if 60 <= edges[-1]:
        ax.set_xticks([60])
    else:
        ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis="x", direction="in", length=3)


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_ablation(traces, trace_dir: Path, output_path: Path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
    })

    all_totals = []
    for series in traces.values():
        totals = np.zeros(len(series["time"]))
        for tp in TP_ORDER:
            if tp in series:
                totals += series[tp]
        if len(totals) > 0:
            all_totals.append(float(totals.max()))
    y_max = max(all_totals) if all_totals else 1.0
    y_max = max(4.0, np.ceil(y_max / 4.0) * 4.0)

    fig, axes = plt.subplots(3, 2, figsize=(3.45, 1.8), sharex=True)

    legend_labels = [tp for tp in ["TP-1", "TP-2", "TP-4"] if any(tp in series for series in traces.values())]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=TP_COLORS[label], alpha=0.85) for label in legend_labels]

    for panel_idx, rps in enumerate(PANEL_LAYOUT):
        col_idx = panel_idx // 3
        row_idx = panel_idx % 3
        ax = axes[row_idx, col_idx]

        if rps is None:
            draw_arrival_axis(ax, trace_dir, ARRIVAL_RPS + 1)
            ax.set_ylabel("RPS")
            continue

        if rps not in traces:
            ax.set_title(f"RPS {rps}")
            ax.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        draw_config_axis(ax, traces[rps], rps, y_max)

    fig.text(0.055, 0.36, "Num Instances", rotation=90, va="center", ha="center")
    # fig.supxlabel("Time (min)", x=0.55,y=-0.03)
    fig.supxlabel("Time", x=0.55,y=0.02)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            bbox_to_anchor=(0.55, 1.02),
            frameon=True,
            handlelength=1.2,
            columnspacing=0.8,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.3,
        )

    fig.subplots_adjust(wspace=0.18, hspace=0.34, top=0.84, left=0.14, right=0.98, bottom=0.1)
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing per-QPS simulation output folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the ablation figure.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=DEFAULT_TRACE_DIR,
        help="Directory containing the BurstGPT arrival traces.",
    )
    parser.add_argument(
        "--rps-values",
        type=int,
        nargs="+",
        default=DEFAULT_RPS_VALUES,
        help="QPS values to include in the figure.",
    )
    args = parser.parse_args()

    traces = {}
    for rps in args.rps_values:
        try:
            traces[rps] = load_reconfig_trace(resolve_config_history(args.input_root, rps))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    plot_ablation(traces, args.trace_dir, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()