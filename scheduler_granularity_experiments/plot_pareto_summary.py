#!/usr/bin/env python3

"""Plot a Pareto-style summary from a final_summary.txt file.

The plot uses x = accuracy and y = SLO attainment.
Points are grouped by setup:
  - all_features vs no_confidence_features
  - sync_true vs sync_false

Each group gets a distinct base color. Within a group, larger x-values are
drawn with darker shades of that base color.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


TABLE_ROW_RE = re.compile(
    r"^(?P<experiment>[^|]+?)\s*\|\s*"
    r"(?P<slo>[0-9.]+)\s*\|\s*"
    r"(?P<acc>[0-9.]+)\s*\|\s*"
    r"(?P<avg_rt>[0-9.]+)\s*\|\s*"
    r"(?P<sched>[0-9.]+)\s*\|\s*"
    r"(?P<docs>[0-9]+)\s*\|\s*"
    r"(?P<pred_cnt>[0-9.]+)\s*\|\s*"
    r"(?P<pred_samp>[0-9.]+)\s*\|\s*"
    r"(?P<conf>[0-9.]+)\s*$"
)


@dataclass(frozen=True)
class ExperimentPoint:
    experiment: str
    slo: float
    acc: float
    delta: int
    setup: str


def parse_experiment_name(name: str) -> tuple[str, str, int]:
    parts = name.strip().split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected experiment name: {name}")

    try:
        delta_index = parts.index("delta")
    except ValueError as exc:
        raise ValueError(f"Unexpected experiment name: {name}") from exc

    setup = "_".join(parts[:delta_index])
    sync = "_".join(parts[-2:])
    delta_part = parts[delta_index + 1]
    if not delta_part.isdigit():
        raise ValueError(f"Unexpected experiment name: {name}")

    return setup, sync, int(delta_part)


def parse_summary(path: Path) -> list[ExperimentPoint]:
    points: list[ExperimentPoint] = []
    for raw_line in path.read_text().splitlines():
        match = TABLE_ROW_RE.match(raw_line)
        if not match:
            continue

        experiment = match.group("experiment").strip()
        setup, sync, delta = parse_experiment_name(experiment)
        points.append(
            ExperimentPoint(
                experiment=experiment,
                slo=float(match.group("slo")),
                acc=float(match.group("acc")),
                delta=delta,
                setup=f"{setup}_{sync}",
            )
        )

    return points


def blend_with_black(color: str, darkness: float) -> tuple[float, float, float]:
    """Blend a color with black.

    darkness should be in [0, 1], where 0 is the lightest shade and 1 is the
    original base color.
    """

    darkness = min(max(darkness, 0.0), 1.0)
    r, g, b = to_rgb(color)
    return (r * darkness, g * darkness, b * darkness)


def build_color_map(points: list[ExperimentPoint]) -> dict[str, tuple[str, list[ExperimentPoint]]]:
    groups: dict[str, list[ExperimentPoint]] = {}
    for point in points:
        groups.setdefault(point.setup, []).append(point)

    # Stable palette with four distinct base colors.
    base_colors = {
        "all_features_sync_false": "#1f77b4",
        "all_features_sync_true": "#d62728",
        "no_confidence_features_sync_false": "#2ca02c",
        "no_confidence_features_sync_true": "#ff7f0e",
    }

    missing = sorted(set(groups) - set(base_colors))
    if missing:
        raise ValueError(f"Unexpected setup groups: {missing}")

    return {setup: (base_colors[setup], groups[setup]) for setup in groups}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a Pareto-style curve from a final_summary.txt file."
    )
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to the averaged final_summary.txt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pareto_summary.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pareto Curve: Accuracy vs SLO",
        help="Plot title",
    )
    args = parser.parse_args()

    points = parse_summary(args.summary)
    if not points:
        raise SystemExit(f"No experiment rows found in {args.summary}")

    grouped = build_color_map(points)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    ax.set_title(args.title)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("SLO attainment")
    ax.grid(True, linestyle="--", alpha=0.25)

    for setup, (base_color, group_points) in grouped.items():
        ordered = sorted(group_points, key=lambda item: item.acc)
        min_x = min(point.acc for point in ordered)
        max_x = max(point.acc for point in ordered)
        x_span = max(max_x - min_x, 1e-9)

        line_x = [point.acc for point in ordered]
        line_y = [point.slo for point in ordered]
        ax.plot(line_x, line_y, color=blend_with_black(base_color, 0.85), linewidth=1.5, alpha=0.8)

        for point in ordered:
            darkness = 0.35 + 0.65 * ((point.acc - min_x) / x_span)
            ax.scatter(
                point.acc,
                point.slo,
                s=70,
                color=blend_with_black(base_color, darkness),
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.annotate(
                str(point.delta),
                (point.acc, point.slo),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="black",
            )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="all_features_sync_false",
                   markerfacecolor="#1f77b4", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="all_features_sync_true",
                   markerfacecolor="#d62728", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="no_confidence_features_sync_false",
                   markerfacecolor="#2ca02c", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="no_confidence_features_sync_true",
                   markerfacecolor="#ff7f0e", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()