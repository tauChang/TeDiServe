#!/usr/bin/env python3

"""Average per-experiment metrics across multiple final_summary.txt files.

The script expects each input file to contain a table like:

    experiment | slo | acc | avg_rt(s) | sched(ms) | docs | pred_cnt | pred_samp | conf

It computes the average for each metric across every experiment that appears
in at least one provided file. For each experiment, each metric is averaged
only across the files where that experiment is present.

The merged summary is written to an output final_summary.txt file.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TABLE_ROW_RE = re.compile(
    r"^(?P<experiment>[^|]+?)\s*\|\s*"
    r"(?P<slo>[0-9.]+)\s*\|\s*"
    r"(?P<acc>[0-9.]+)\s*\|\s*"
    r"(?P<avg_rt>[0-9.]+)\s*\|\s*"
    r"(?P<sched>[0-9.]+)\s*\|\s*"
    r"(?P<docs>[0-9]+)\s*\|\s*"
    r"(?P<pred_cnt>[0-9]+)\s*\|\s*"
    r"(?P<pred_samp>[0-9]+)\s*\|\s*"
    r"(?P<conf>[0-9.]+)\s*$"
)


@dataclass(frozen=True)
class ExperimentMetrics:
    slo: float
    acc: float
    avg_rt: float
    sched: float
    docs: int
    pred_cnt: int
    pred_samp: int
    conf: float


def parse_summary(path: Path) -> dict[str, ExperimentMetrics]:
    """Parse one summary file into a mapping of experiment -> metrics."""
    experiments: dict[str, ExperimentMetrics] = {}

    for raw_line in path.read_text().splitlines():
        match = TABLE_ROW_RE.match(raw_line)
        if not match:
            continue

        experiment = match.group("experiment").strip()
        experiments[experiment] = ExperimentMetrics(
            slo=float(match.group("slo")),
            acc=float(match.group("acc")),
            avg_rt=float(match.group("avg_rt")),
            sched=float(match.group("sched")),
            docs=int(match.group("docs")),
            pred_cnt=int(match.group("pred_cnt")),
            pred_samp=int(match.group("pred_samp")),
            conf=float(match.group("conf")),
        )

    return experiments


def average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average experiments across multiple final_summary.txt files."
    )
    # parser.add_argument(
    #     "summaries",
    #     nargs="+",
    #     type=Path,
    #     help="Paths to final_summary.txt files",
    # )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_summary.txt"),
        help="Path to write the averaged final summary",
    )
    args = parser.parse_args()

    # parsed_summaries = [parse_summary(path) for path in args.summaries]
    paths = [
        Path("/u/tchang85/dllm/sbatch_experiment_dir/20260420/013533/final_summary.txt"),
        Path("/u/tchang85/dllm/sbatch_experiment_dir/20260420/083407/final_summary.txt"),
        Path("/u/tchang85/dllm/sbatch_experiment_dir/20260420/093612/final_summary.txt"),
        Path("/u/tchang85/dllm/sbatch_experiment_dir/20260420/134151/final_summary.txt"),
    ]
    parsed_summaries = [parse_summary(path) for path in paths]
    experiment_order: list[str] = []
    seen: set[str] = set()
    for summary in parsed_summaries:
        for experiment in summary:
            if experiment not in seen:
                seen.add(experiment)
                experiment_order.append(experiment)

    if not experiment_order:
        raise SystemExit("No experiments were found in the provided summary files.")

    lines: list[str] = []
    lines.append(f"=== AVERAGED SUMMARY ACROSS {len(parsed_summaries)} FILES ===")
    lines.append(f"experiments: {len(experiment_order)}")
    lines.append("")
    lines.append(
        "experiment                     |    slo |    acc | avg_rt(s) |  sched(ms) |   docs | pred_cnt | pred_samp |   conf"
    )
    lines.append("-" * 120)

    for experiment in experiment_order:
        metrics = [summary[experiment] for summary in parsed_summaries if experiment in summary]
        lines.append(
            f"{experiment:30} | "
            f"{average(metric.slo for metric in metrics):6.3f} | "
            f"{average(metric.acc for metric in metrics):6.3f} | "
            f"{average(metric.avg_rt for metric in metrics):9.3f} | "
            f"{average(metric.sched for metric in metrics):10.4f} | "
            f"{average(metric.docs for metric in metrics):6.0f} | "
            f"{average(metric.pred_cnt for metric in metrics):8.1f} | "
            f"{average(metric.pred_samp for metric in metrics):9.1f} | "
            f"{average(metric.conf for metric in metrics):6.3f}"
        )

    output_text = "\n".join(lines) + "\n"
    args.output.write_text(output_text)
    print(output_text, end="")
    print(f"Wrote averaged summary to {args.output}")


if __name__ == "__main__":
    main()