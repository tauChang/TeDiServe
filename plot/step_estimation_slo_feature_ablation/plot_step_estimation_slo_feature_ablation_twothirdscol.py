from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


OUTPUT_DIR = Path(__file__).resolve().parent


OUTPUT_STEM = "step_estimation_slo_feature_ablation_1x2_twothirdscol"
LOCAL_SUMMARY_PATH = OUTPUT_DIR / "results.txt"
BASE_SLO_SECONDS = 1.045
FONT_SIZE = 9.5
LINE_WIDTH = 1.2
MARKER_SIZE = 9
FIG_WIDTH = 2.30
FIG_HEIGHT = 1.6

BASELINE_ORDER = ["oracle", "one_shot", "no_conf", "all_features"]
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
BASE_COLORS = {
    "oracle": "#1f77b4",
    "all_features": "#c43c39",
    "no_conf": "#f28e2b",
    "one_shot": "#2ca02c",
}
METRIC_AXIS_CONFIG = {
    "slo_attainment": {
        "ylim": (0.53, 1.0),
        "yticks": [0.6, 0.7, 0.8, 0.9, 1.0],
    }
}
PERCENT_INTEGER_FORMATTER = FuncFormatter(lambda value, _: f"{value * 100:.0f}")

ABLATION_ROWS = [
    ("Output Len", 0.7502),
    ("+ Progress", 0.3515),
    ("+ Conf.", 0.1926),
    ("Oracle", 0.0),
]

LEGEND_ORDER = ["oracle", "one_shot", "no_conf", "all_features"]
LEGEND_LABELS = {
    "oracle": "Oracle",
    "all_features": "Online (Progress + Conf. Features)",
    "no_conf": "Online (Progress Features)",
    "one_shot": "One-shot",
}
FEATURE_LABEL_TO_BASELINE = {
    "Output Len": "one_shot",
    "+ Progress": "no_conf",
    "+ Conf.": "all_features",
    "Oracle": "oracle",
}


def parse_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("-") or line.startswith("experiment"):
                continue
            if "|" not in line:
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 9:
                continue

            experiment = parts[0]
            try:
                slo_attainment = float(parts[1])
                accuracy = float(parts[2])
                confidence = float(parts[8])
            except ValueError:
                continue

            rows.append(
                {
                    "experiment": experiment,
                    "slo_attainment": slo_attainment,
                    "accuracy": accuracy,
                    "confidence": confidence,
                }
            )
    return rows


def parse_numeric_suffix(experiment: str, key: str) -> float | None:
    match = re.search(rf"{key}_(\d+(?:_\d+)?)", experiment)
    if not match:
        return None
    return float(match.group(1).replace("_", "."))


def parse_layout(experiment: str) -> str | None:
    if re.search(r"slo_\d+(?:_\d+)?_rps_\d", experiment):
        return "rps"
    if re.search(r"rps_\d+(?:_\d+)?_slo_\d", experiment):
        return "slo"
    return None


def model_key(experiment: str) -> str | None:
    if experiment.startswith("oracle_"):
        return "oracle"
    if experiment.startswith("all_features_"):
        return "all_features"
    if experiment.startswith("no_conf_no_progress_"):
        return "one_shot"
    if experiment.startswith("no_conf_"):
        return "no_conf"
    return None


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in rows:
        experiment = str(row["experiment"])
        baseline = model_key(experiment)
        layout = parse_layout(experiment)
        rps = parse_numeric_suffix(experiment, "rps")
        slo = parse_numeric_suffix(experiment, "slo")
        if baseline is None or layout is None or rps is None or slo is None:
            continue

        normalized.append(
            {
                **row,
                "baseline": baseline,
                "layout": layout,
                "rps": rps,
                "rps_per_gpu": rps / 4.0,
                "slo": slo,
                "slo_multiple": slo / BASE_SLO_SECONDS,
            }
        )
    return normalized


def format_multiple_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}x"
    return f"{value:.2f}".rstrip("0").rstrip(".") + "x"


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


def plot_feature_ablation_panel(ax) -> None:
    labels = [row[0] for row in ABLATION_ROWS]
    values = [row[1] for row in ABLATION_ROWS]
    colors = [BASE_COLORS[FEATURE_LABEL_TO_BASELINE[row[0]]] for row in ABLATION_ROWS]
    positions = np.arange(len(labels))

    bars = ax.barh(positions, values, color=colors, height=0.85)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Pred. Error (%)", labelpad=0.5, x=0.45)
    ax.tick_params(axis="x", direction="in", pad=1.2, length=1.5)
    ax.tick_params(axis="y", length=0, pad=1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 0.78)
    ax.set_xticks([0, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0", "25", "50", "75"])

    max_value = max(values)
    for bar, value in zip(bars, values):
        y_center = bar.get_y() + bar.get_height() / 2
        if value < 0.7:
            x_text = value + max_value * 0.012 if value > 0 else max_value * 0.02
            ax.text(
                x_text,
                y_center,
                f"{value * 100:.1f}",
                va="center",
                ha="left",
                fontsize=7,
                fontweight="bold",
            )
        else:
            ax.text(
                value - 0.018,
                y_center - 0.25,
                f"{value * 100:.1f}",
                va="top",
                ha="right",
                fontsize=7,
                color="white",
                fontweight="bold",
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
            "xtick.labelsize": FONT_SIZE - 0.4,
            "ytick.labelsize": FONT_SIZE - 0.4,
            "legend.fontsize": FONT_SIZE - 0.4,
            "font.family": "serif",
            "lines.linewidth": LINE_WIDTH,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    metric_cfg = METRIC_AXIS_CONFIG["slo_attainment"]
    plot_metric_compact(
        axes[0],
        slo_rows,
        "slo_attainment",
        slo_x_values,
        slo_tick_positions,
        slo_tick_labels,
        (3.05, 1.45),
    )
    axes[0].set_ylim(*metric_cfg["ylim"])
    axes[0].set_yticks(metric_cfg["yticks"])
    axes[0].set_ylabel("SLO Att. (%)", labelpad=0.8)
    axes[0].set_xlabel("SLO Multiple", labelpad=0.5)
    axes[0].tick_params(axis="both", direction="in", pad=1.2, length=1.2)
    axes[0].yaxis.set_major_formatter(PERCENT_INTEGER_FORMATTER)

    plot_feature_ablation_panel(axes[1])

    legend_handles = [
        Patch(facecolor=BASE_COLORS[key], edgecolor="none", label=LEGEND_LABELS[key])
        for key in LEGEND_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.48, 1.02),
        columnspacing=0.8,
        handlelength=1,
        handletextpad=0.3,
        labelspacing=0.18,
        fontsize=FONT_SIZE - 1.0,
    )

    fig.subplots_adjust(left=-0.01, right=1.03, bottom=0.24, top=0.72, wspace=0.42)

    right_pos = axes[1].get_position()
    label_room = 0.07
    axes[1].set_position(
        [
            right_pos.x0 + label_room+0.1,
            right_pos.y0,
            right_pos.width - label_room,
            right_pos.height,
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_dir / f"{OUTPUT_STEM}.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)

    print(f"Saved plot: {out_dir / f'{OUTPUT_STEM}.png'}")
    print(f"Saved plot: {out_dir / f'{OUTPUT_STEM}.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=LOCAL_SUMMARY_PATH,
        help="Path to the summary results table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the figure will be written.",
    )
    args = parser.parse_args()
    main(args.summary, args.out_dir)