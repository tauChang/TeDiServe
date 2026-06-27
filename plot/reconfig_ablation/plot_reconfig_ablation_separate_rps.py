import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_reconfig_ablation import (
    ARRIVAL_RPS,
    ARRIVAL_WINDOW_MIN,
    DEFAULT_INPUT_ROOT,
    DEFAULT_TRACE_DIR,
    TP_COLORS,
    TP_ORDER,
    compute_qps_from_arrivals,
    load_arrivals,
    load_reconfig_trace,
    resolve_config_history,
    save_figure,
)


DEFAULT_OUTPUT = Path(
    "/u/tchang85/dllm/plot/reconfig_ablation/reconfig_ablation_separate_rps.png"
)
DEFAULT_RPS_VALUES = [2, 4, 6, 8]
PANEL_RPS_ORDER = [2, 6, 4, 8]
TIME_TICKS_MIN = [0, 30, 60, 90, 120]


def compute_y_max(traces):
    all_totals = []
    for series in traces.values():
        totals = np.zeros(len(series["time"]))
        for tp in TP_ORDER:
            if tp in series:
                totals += series[tp]
        if len(totals) > 0:
            all_totals.append(float(totals.max()))

    y_max = max(all_totals) if all_totals else 1.0
    return max(4.0, np.ceil(y_max / 4.0) * 4.0)


def draw_arrival_axis(ax, trace_dir: Path):
    arrivals = load_arrivals(trace_dir / f"burstgpt_2hrs_{ARRIVAL_RPS}qps.txt")
    times_min, qps = compute_qps_from_arrivals(arrivals, ARRIVAL_WINDOW_MIN)
    ax.plot(times_min, qps, color="black")
    # ax.set_title("Arrival Rate", pad=2)
    ax.set_xlim(0, TIME_TICKS_MIN[-1])
    ax.set_xticks(TIME_TICKS_MIN)
    ax.set_ylim(0, ARRIVAL_RPS + 1)
    # set y label "RPS"
    ax.set_ylabel("RPS", labelpad=1)
    ax.set_yticks([])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", direction="in", length=3)


def draw_config_axis(ax, series, rps: int, y_max: float, show_xlabel: bool, show_yticks: bool):
    cfg_time = series["time"]
    if len(cfg_time) == 0:
        ax.set_title(f"Peak RPS {rps}", pad=2)
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

    ax.set_title(f"Peak RPS {rps}", pad=2)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, y_max)
    if show_yticks:
        ax.set_yticks([tick for tick in [0, 8, 16] if tick <= y_max])
    else:
        ax.set_yticks([])
    ax.set_xlim(0, TIME_TICKS_MIN[-1])
    ax.set_xticks(TIME_TICKS_MIN)
    if show_xlabel:
        ax.set_xticklabels([str(tick) for tick in TIME_TICKS_MIN])
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", direction="in", length=3)


def plot_ablation(traces, trace_dir: Path, output_path: Path):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.2,
        }
    )

    y_max = compute_y_max(traces)

    fig = plt.figure(figsize=(3.45, 1.8))
    grid = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    fig.subplots_adjust(wspace=0.12, hspace=0.45, top=0.58, left=0.16, right=0.98, bottom=0.12)

    left_bbox = axes[0].get_position()
    right_bbox = axes[1].get_position()
    arrival_width = left_bbox.width
    arrival_height = left_bbox.height
    arrival_x0 = 0.5 * (left_bbox.x0 + right_bbox.x1) - 0.5 * arrival_width
    arrival_y0 = 0.72
    ax_arrival = fig.add_axes([arrival_x0, arrival_y0, arrival_width, arrival_height])

    draw_arrival_axis(ax_arrival, trace_dir)

    legend_labels = [
        tp for tp in ["TP-1", "TP-2", "TP-4"] if any(tp in series for series in traces.values())
    ]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=TP_COLORS[label], alpha=0.85)
        for label in legend_labels
    ]

    for ax, rps in zip(axes, PANEL_RPS_ORDER):
        if rps not in traces:
            ax.set_title(f"Peak RPS {rps}")
            ax.text(0.5, 0.5, "Missing data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        show_xlabel = rps in [4, 8]
        show_yticks = rps in [2, 4]
        draw_config_axis(
            ax,
            traces[rps],
            rps,
            y_max,
            show_xlabel=show_xlabel,
            show_yticks=show_yticks,
        )

    fig.text(0.095, 0.35, "Num Instances", rotation=90, va="center", ha="center")
    fig.supxlabel("Time (min)", y=-0.02, x=0.57)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            ncol=1,
            bbox_to_anchor=(arrival_x0 + arrival_width + 0.08, arrival_y0 + 0.5 * arrival_height-0.04),
            frameon=True,
            handlelength=1.2,
            columnspacing=0.8,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.3,
        )
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
    args = parser.parse_args()

    traces = {}
    for rps in DEFAULT_RPS_VALUES:
        try:
            traces[rps] = load_reconfig_trace(resolve_config_history(args.input_root, rps))
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    plot_ablation(traces, args.trace_dir, args.output)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()