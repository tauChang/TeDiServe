from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


BASE_SLO_SECONDS = 1.045
FONT_SIZE = 20
LINE_WIDTH = 3.5
MARKER_SIZE = 140

BASELINE_ORDER = [
    "oracle",
    "one_shot",
    "no_conf",
    "all_features",
]

BASE_COLORS = {
    "oracle": "#1f77b4",
    "all_features": "#c43c39",
    "no_conf": "#f28e2b",
    "one_shot": "#2ca02c",
}

LEGEND_LABELS = {
    "oracle": "Oracle",
    "all_features": "Online (Progress + Confidence Features)",
    "no_conf": "Online (Progress Features)",
    "one_shot": "One-shot",
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


def metric_limits(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    if key == "slo_attainment":
        return 0.5, 1.0
    if key == "confidence":
        return 0.6, 0.9

    values = [float(row[key]) for row in rows]
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 0.02)
    padding = 0.12 * span
    return min_value - padding, max_value + padding


def format_rps_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_multiple_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}x"
    return f"{value:.2f}".rstrip("0").rstrip(".") + "x"


def plot_metric(
    ax,
    rows: list[dict[str, object]],
    x_key: str,
    y_key: str,
    x_values: list[float],
    tick_labels: list[str],
    reverse_x: bool = False,
    tick_positions: list[float] | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    grouped: dict[str, list[dict[str, object]]] = {key: [] for key in BASELINE_ORDER}
    for row in rows:
        grouped[str(row["baseline"])].append(row)

    for baseline in BASELINE_ORDER:
        series = grouped[baseline]
        if not series:
            continue

        series.sort(key=lambda item: x_values.index(float(item[x_key])))
        x = [float(item[x_key]) for item in series]
        y = [float(item[y_key]) for item in series]

        ax.plot(
            x,
            y,
            color=BASE_COLORS[baseline],
            linewidth=LINE_WIDTH,
            zorder=3,
        )
        ax.scatter(
            x,
            y,
            color=BASE_COLORS[baseline],
            edgecolor=BASE_COLORS[baseline],
            s=MARKER_SIZE,
            zorder=4,
        )

    axis_tick_positions = tick_positions if tick_positions is not None else x_values
    ax.set_xticks(axis_tick_positions)
    ax.set_xticklabels(tick_labels)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    elif reverse_x:
        ax.set_xlim(max(x_values), min(x_values))
    ax.grid(True, linestyle="--", alpha=0.28)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def main(summary_path: Path, out_dir: Path) -> None:
    rows = normalize_rows(parse_summary(summary_path))

    rps_rows = [row for row in rows if row["layout"] == "rps"]
    slo_rows = [row for row in rows if row["layout"] == "slo"]
    if not rps_rows or not slo_rows:
        raise ValueError("Expected both RPS-sweep and SLO-sweep rows in the summary file.")

    rps_x_values = sorted({float(row["rps_per_gpu"]) for row in rps_rows})
    slo_x_values = sorted({float(row["slo_multiple"]) for row in slo_rows})
    slo_tick_positions = [1.5, 2.0, 2.5, 3.0, 4, 5]

    rps_tick_labels = [format_rps_tick(value) for value in rps_x_values]
    slo_tick_labels = [format_multiple_tick(value) for value in slo_tick_positions]

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE - 2,
            "font.family": "serif",
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0), sharex="col")

    metric_specs = [
        ("slo_attainment", "SLO Attainment"),
        ("confidence", "Conf. / Token"),
        ("accuracy", "Accuracy"),
    ]

    for row_idx, (metric_key, ylabel) in enumerate(metric_specs):
        plot_metric(
            axes[row_idx, 0],
            rps_rows,
            "rps_per_gpu",
            metric_key,
            rps_x_values,
            rps_tick_labels,
            reverse_x=False,
        )
        plot_metric(
            axes[row_idx, 1],
            slo_rows,
            "slo_multiple",
            metric_key,
            slo_x_values,
            slo_tick_labels,
            reverse_x=True,
            tick_positions=slo_tick_positions,
            x_limits=(3.0, 1.5),
        )

        ymin, ymax = metric_limits(rows, metric_key)
        axes[row_idx, 0].set_ylim(ymin, ymax)
        axes[row_idx, 1].set_ylim(ymin, ymax)
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=FONT_SIZE)

    axes[0, 0].set_title("Varying RPS", fontsize=FONT_SIZE)
    axes[0, 1].set_title("Varying SLO", fontsize=FONT_SIZE)
    axes[2, 0].set_xlabel("RPS / GPU", fontsize=FONT_SIZE)
    axes[2, 1].set_xlabel("SLO Multiple", fontsize=FONT_SIZE)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.tick_params(labelbottom=False)
    for ax in [axes[0, 1], axes[1, 1], axes[2, 1]]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BASE_COLORS[key],
            lw=LINE_WIDTH,
            marker="o",
            markersize=10,
            label=LEGEND_LABELS[key],
        )
        for key in BASELINE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.995),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.subplots_adjust(wspace=0.10, hspace=0.10)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "step_estimation_ablation_3x2.png", dpi=220)
    fig.savefig(out_dir / "step_estimation_ablation_3x2.pdf")
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