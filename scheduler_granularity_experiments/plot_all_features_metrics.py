#!/usr/bin/env python3

"""Plot feature-set metrics across delta values, split by sync mode.

The figure uses a 2x2 layout with one panel each for:
  - SLO attainment
  - scheduling overhead
  - average confidence
  - accuracy

Only all_features rows for deltas {1,2,4,8,16,24,32} are plotted, plus the
single no_confidence_features_delta_1000_sync_true row. Each panel compares
sync_false vs sync_true with consistent line styles.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


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

TARGET_DELTAS = [1, 2, 4, 8, 16, 24, 32, 1000]
SYNC_ORDER = ["sync_false", "sync_true"]
DELTA_POSITIONS = {delta: idx for idx, delta in enumerate(TARGET_DELTAS)}


@dataclass(frozen=True)
class ExperimentMetrics:
    experiment: str
    setup: str
    sync: str
    delta: int
    slo: float
    acc: float
    sched: float
    conf: float


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


def parse_summary(path: Path) -> list[ExperimentMetrics]:
    points: list[ExperimentMetrics] = []
    for raw_line in path.read_text().splitlines():
        match = TABLE_ROW_RE.match(raw_line)
        if not match:
            continue

        experiment = match.group("experiment").strip()
        setup, sync, delta = parse_experiment_name(experiment)
        if setup == "all_features" and delta == 1000:
            continue
        if setup != "all_features" and not (
            setup == "no_confidence_features" and sync == "sync_true" and delta == 1000
        ):
            continue
        if delta not in TARGET_DELTAS:
            continue

        points.append(
            ExperimentMetrics(
                experiment=experiment,
                setup=setup,
                sync=sync,
                delta=delta,
                slo=float(match.group("slo")),
                acc=float(match.group("acc")),
                sched=float(match.group("sched")),
                conf=float(match.group("conf")),
            )
        )

    return points


def group_points(points: list[ExperimentMetrics]) -> dict[str, dict[int, ExperimentMetrics]]:
    grouped: dict[str, dict[int, ExperimentMetrics]] = {sync: {} for sync in SYNC_ORDER}
    for point in points:
        if point.sync not in grouped:
            raise ValueError(f"Unexpected sync group: {point.sync}")
        grouped[point.sync][point.delta] = point
    return grouped


def plot_metric(
    ax,
    grouped: dict[str, dict[int, ExperimentMetrics]],
    metric_name: str,
    ylabel: str,
) -> None:
    for sync, label, linestyle, marker in [
        ("sync_true", "sync", "-", "o"),
        ("sync_false", "async", "--", "s"),
    ]:
        values = grouped[sync]
        xs = [DELTA_POSITIONS[delta] for delta in TARGET_DELTAS if delta in values]
        if metric_name == "sched":
            ys = [100.0 * getattr(values[delta], metric_name) / 30.0 for delta in TARGET_DELTAS if delta in values]
        else:
            ys = [getattr(values[delta], metric_name) for delta in TARGET_DELTAS if delta in values]
        ax.plot(
            xs,
            ys,
            linestyle=linestyle,
            marker=marker,
            linewidth=2.6,
            markersize=7,
            label=label,
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(list(DELTA_POSITIONS.values()))
    ax.set_xticklabels([str(delta) for delta in TARGET_DELTAS])
    ax.tick_params(axis="both")
    ax.grid(True, linestyle="--", alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot all_features metrics across delta values, split by sync mode."
    )
    parser.add_argument(
        "summary",
        type=Path,
        help="Path to the averaged final_summary.txt file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("all_features_metrics.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="All-Features Metrics vs Delta",
        help="Plot title",
    )
    args = parser.parse_args()

    points = parse_summary(args.summary)
    if not points:
        raise SystemExit(f"No target setup rows found in {args.summary}")

    grouped = group_points(points)
    expected_points = {
        "sync_false": {delta for delta in TARGET_DELTAS if delta != 1000},
        "sync_true": set(TARGET_DELTAS),
    }
    missing = [
        sync
        for sync, deltas in expected_points.items()
        if set(grouped[sync]) != deltas
    ]
    if missing:
        raise SystemExit(f"Missing deltas for sync groups: {missing}")

    plt.rcParams.update(
        {
            "axes.titlesize": 24,
            "axes.labelsize": 32,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
            "legend.fontsize": 32,
        }
    )

    fig, axes = plt.subplots(4, 1, figsize=(15, 24), dpi=200, sharex=True)
    # fig.suptitle(args.title, fontsize=26, y=0.995)

    panels = [
        (axes[0], "slo", "SLO Attainment"),
        (axes[1], "sched", "Scheduling Overhead (%)"),
        (axes[2], "conf", "Confidence Thresh Used"),
        (axes[3], "acc", "Accuracy"),
    ]

    for ax, metric_name, ylabel in panels:
        plot_metric(ax, grouped, metric_name, ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=32,
    )

    fig.supxlabel("Update Granularity", fontsize=32)

    for ax in axes:
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 0.84, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()