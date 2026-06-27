from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

from plot_step_estimation_ablation import (
    BASELINE_ORDER,
    BASE_COLORS,
    LEGEND_LABELS,
    format_multiple_tick,
    normalize_rows,
    parse_summary,
)


FONT_SIZE = 7
LINE_WIDTH = 1.5
MARKER_SIZE = 12

PERCENT_INTEGER_FORMATTER = FuncFormatter(lambda value, _: f"{value * 100:.0f}")

METRIC_AXIS_CONFIG = {
    "slo_attainment": {
        "title": "SLO Attain (%)",
        "ylim": (0.53, 1.0),
        "yticks": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "confidence": {
        "title": "Conf. / Token",
        "ylim": (0.58, 0.84),
        "yticks": [0.6, 0.7, 0.8],
    },
    "accuracy": {
        "title": "Accuracy",
        "ylim": (0.67, 0.765),
        "yticks": [0.68, 0.70, 0.72, 0.74, 0.76],
    },
}

LEGEND_ORDER = ["oracle", "one_shot", "no_conf", "all_features"]
BASELINE_ZORDER = {
    "one_shot": 1,
    "no_conf": 2,
    "all_features": 3,
    "oracle": 4,
}
BASELINE_MARKERS = {
    "oracle": "o",
    "one_shot": "s",
    "no_conf": "^",
    "all_features": "D",
}


def plot_metric_compact(
    ax,
    rows: list[dict[str, object]],
    y_key: str,
    x_values: list[float],
    tick_positions: list[float],
    tick_labels: list[str],
    x_limits: tuple[float, float],
) -> None:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in BASELINE_ORDER}
    for row in rows:
        grouped[str(row["baseline"])].append(row)

    for baseline in BASELINE_ORDER:
        series = grouped[baseline]
        if not series:
            continue

        baseline_zorder = BASELINE_ZORDER[baseline]

        series.sort(key=lambda item: x_values.index(float(item["slo_multiple"])))
        x = [float(item["slo_multiple"]) for item in series]
        y = [float(item[y_key]) for item in series]

        ax.plot(x, y, color=BASE_COLORS[baseline], linewidth=LINE_WIDTH, zorder=2 * baseline_zorder)
        ax.scatter(
            x,
            y,
            color=BASE_COLORS[baseline],
            edgecolor=BASE_COLORS[baseline],
            s=MARKER_SIZE,
            marker=BASELINE_MARKERS[baseline],
            zorder=2 * baseline_zorder + 0.5,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(x_limits)
    ax.grid(True, linestyle="--", alpha=0.28)


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
            "axes.titlesize": FONT_SIZE + 1,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 0.5,
            "ytick.labelsize": FONT_SIZE - 0.5,
            "legend.fontsize": FONT_SIZE,
            "font.family": "serif",
            "lines.linewidth": LINE_WIDTH,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.55))

    metric_order = ["slo_attainment", "confidence", "accuracy"]

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
        ax.set_title(metric_cfg["title"], pad=2)
        ax.tick_params(axis="both", direction="in", pad=1.0)
        if metric_key in {"slo_attainment", "accuracy"}:
            ax.yaxis.set_major_formatter(PERCENT_INTEGER_FORMATTER)
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    fig.supxlabel("SLO Multiple", y=0.13, fontsize=FONT_SIZE)

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
        bbox_to_anchor=(0.5, 1.05),
        columnspacing=1.0,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.28, top=0.68, wspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "step_estimation_ablation_slo_only_1x3.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / "step_estimation_ablation_slo_only_1x3.pdf", bbox_inches="tight", pad_inches=0.02)
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