from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_step_estimation_ablation import format_multiple_tick, normalize_rows, parse_summary
from plot_step_estimation_ablation_slo_only import (
    BASELINE_MARKERS,
    BASE_COLORS,
    FONT_SIZE,
    LEGEND_ORDER,
    LEGEND_LABELS,
    LINE_WIDTH,
    METRIC_AXIS_CONFIG,
    PERCENT_INTEGER_FORMATTER,
    plot_metric_compact,
)


def main(summary_path: Path, out_dir: Path) -> None:
    rows = normalize_rows(parse_summary(summary_path))
    slo_rows = [row for row in rows if row["layout"] == "slo"]
    if not slo_rows:
        raise ValueError("Expected SLO-sweep rows in the summary file.")

    slo_x_values = sorted({float(row["slo_multiple"]) for row in slo_rows})
    slo_tick_positions = [1.5, 2.0, 2.5, 3.0]
    slo_tick_labels = [format_multiple_tick(value) for value in slo_tick_positions]

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 0.5,
            "ytick.labelsize": FONT_SIZE - 0.5,
            "legend.fontsize": FONT_SIZE,
            "font.family": "serif",
            "lines.linewidth": LINE_WIDTH,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.2, 1.55))

    metric_order = ["slo_attainment", "accuracy"]
    for ax, metric_key in zip(axes, metric_order):
        metric_cfg = METRIC_AXIS_CONFIG[metric_key]
        plot_metric_compact(
            ax,
            slo_rows,
            metric_key,
            slo_x_values,
            slo_tick_positions,
            slo_tick_labels,
            (3.05, 1.45),
        )
        ax.set_ylim(*metric_cfg["ylim"])
        ax.set_yticks(metric_cfg["yticks"])
        if metric_key == "slo_attainment":
            ax.set_ylabel("SLO Attain (%)")
        else:
            ax.set_ylabel("Accuracy (%)")
        ax.tick_params(axis="both", direction="in", pad=1.0)
        ax.yaxis.set_major_formatter(PERCENT_INTEGER_FORMATTER)

    fig.supxlabel("SLO Multiple", x=0.55, y=0.13, fontsize=FONT_SIZE)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BASE_COLORS[key],
            lw=LINE_WIDTH,
            marker=BASELINE_MARKERS[key],
            markersize=5.5,
            label=LEGEND_LABELS[key],
        )
        for key in LEGEND_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.97),
        columnspacing=1.0,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.28, top=0.68, wspace=0.30)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "step_estimation_ablation_slo_acc_only_1x2.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "step_estimation_ablation_slo_acc_only_1x2.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results.txt"),
        help="Path to the summary results table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory where the figure will be written.",
    )
    args = parser.parse_args()
    main(args.summary, args.out_dir)