#!/usr/bin/env python3

"""Rerun confidence aggregation and batch summaries for a set of runs.

The script is intended for directories like:

  experiment_dir/20260420/<run_dir>/
  sbatch_experiment_dir/20260420/<batch_dir>/

For each batch directory, it:
  1. loads the saved experiments_config.json
  2. finds the matching run directory for each experiment
  3. reruns analysis/confidence_over_time/plot.py
  4. reruns analysis/experiment_summary/run.py
  5. rebuilds that batch's final_summary.txt from the refreshed logs

This avoids rerunning model inference.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUMMARY_ROW_RE = re.compile(
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

EXPERIMENT_SUMMARY_RE = {
    "slo_attainment": re.compile(r"SLO attainment \(≤.*?s\): ([0-9.]+)"),
    "accuracy": re.compile(r"Accuracy: ([0-9.]+)"),
    "avg_response_time": re.compile(r"Average response time: ([0-9.]+)s"),
    "avg_scheduler_latency": re.compile(r"Average scheduler overhead / latency: ([0-9.]+) ms"),
    "document_number": re.compile(r"Document number: ([0-9]+)"),
    "predict_count": re.compile(r"Predict count: ([0-9]+)"),
    "predict_samples": re.compile(r"Predict samples: ([0-9]+)"),
    "avg_confidence": re.compile(r"Average confidence: ([0-9.]+)"),
}


@dataclass(frozen=True)
class ExperimentSummary:
    slo_attainment: float
    accuracy: float
    avg_response_time: float
    avg_scheduler_latency: float
    document_number: int
    predict_count: int
    predict_samples: int
    avg_confidence: float


def average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def load_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def parse_experiment_summary(path: Path) -> ExperimentSummary:
    text = path.read_text()

    def extract(name: str):
        match = EXPERIMENT_SUMMARY_RE[name].search(text)
        if not match:
            raise ValueError(f"Missing {name} in {path}")
        return match.group(1)

    return ExperimentSummary(
        slo_attainment=float(extract("slo_attainment")),
        accuracy=float(extract("accuracy")),
        avg_response_time=float(extract("avg_response_time")),
        avg_scheduler_latency=float(extract("avg_scheduler_latency")),
        document_number=int(extract("document_number")),
        predict_count=int(extract("predict_count")),
        predict_samples=int(extract("predict_samples")),
        avg_confidence=float(extract("avg_confidence")),
    )


def parse_batch_summary_row(line: str) -> tuple[str, dict[str, float]] | None:
    match = SUMMARY_ROW_RE.match(line)
    if not match:
        return None

    experiment = match.group("experiment").strip()
    return experiment, {
        "slo": float(match.group("slo")),
        "acc": float(match.group("acc")),
        "avg_rt": float(match.group("avg_rt")),
        "sched": float(match.group("sched")),
        "docs": float(match.group("docs")),
        "pred_cnt": float(match.group("pred_cnt")),
        "pred_samp": float(match.group("pred_samp")),
        "conf": float(match.group("conf")),
    }


def run_command(command: list[str], *, cwd: Path | None = None, stdout_path: Path | None = None) -> None:
    stdout_handle = None
    try:
        if stdout_path is not None:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_path.open("w")
        subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
    finally:
        if stdout_handle is not None:
            stdout_handle.close()


def find_latest_run_dir(run_root: Path, experiment_index: int, experiment_name: str) -> Path:
    pattern = f"*_{experiment_index}_{experiment_name}"
    matches = [path for path in run_root.glob(pattern) if path.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find run directory for index={experiment_index}, experiment={experiment_name} under {run_root}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime)


def discover_result_path(run_dir: Path) -> Path:
    candidates = [path for path in run_dir.glob("results/**/*.json") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No result JSON found under {run_dir / 'results'}")
    # Prefer the main evaluation result file if there are multiple JSON files.
    candidates.sort(key=lambda path: ("/samples/" in str(path), len(path.parts), str(path)))
    return candidates[0]


def rerun_run_dir(run_dir: Path, repo_root: Path) -> None:
    summary = load_json(run_dir / "request_plots" / "avg_confidence_stats.json") if (run_dir / "request_plots" / "avg_confidence_stats.json").is_file() else None
    if summary is not None:
        print(f"Rerunning confidence plot for {run_dir.name}")

    plot_cmd = [
        sys.executable,
        str(repo_root / "analysis" / "confidence_over_time" / "plot.py"),
        "--step-data",
        str(run_dir / "step_data.json"),
        "--workload-history",
        str(run_dir / "workload_history.json"),
        "--output-dir",
        str(run_dir / "request_plots"),
    ]
    run_command(plot_cmd, cwd=repo_root)


def rerun_experiment_summary(run_dir: Path, repo_root: Path, slo: float) -> None:
    result_path = discover_result_path(run_dir)
    log_path = run_dir / "logs" / "experiment_summary.log"

    summary_cmd = [
        sys.executable,
        str(repo_root / "analysis" / "experiment_summary" / "run.py"),
        "--result-path",
        str(result_path),
        "--slo",
        str(slo),
        "--scheduler-summary",
        str(run_dir / "profiles" / "scheduler" / "schedule_summary.txt"),
        "--predict-summary",
        str(run_dir / "profiles" / "step_estimator" / "predict_summary.txt"),
        "--confidence-stats",
        str(run_dir / "request_plots" / "avg_confidence_stats.json"),
    ]
    run_command(summary_cmd, cwd=repo_root, stdout_path=log_path)


def rebuild_batch_final_summary(batch_dir: Path, experiment_rows: list[tuple[str, Path]]) -> None:
    summaries: list[tuple[str, ExperimentSummary]] = []
    for experiment_name, run_dir in experiment_rows:
        log_path = run_dir / "logs" / "experiment_summary.log"
        summaries.append((experiment_name, parse_experiment_summary(log_path)))

    lines: list[str] = []
    lines.append("=== FINAL BATCH SUMMARY ===")
    lines.append(f"experiments: {len(summaries)}")
    lines.append("")
    lines.append(
        "experiment                     |    slo |    acc | avg_rt(s) |  sched(ms) |   docs | pred_cnt | pred_samp |   conf"
    )
    lines.append("-" * 120)

    numeric_keys = [
        "slo_attainment",
        "accuracy",
        "avg_response_time",
        "avg_scheduler_latency",
        "document_number",
        "predict_count",
        "predict_samples",
        "avg_confidence",
    ]
    aggregate = {key: [] for key in numeric_keys}

    for experiment_name, metrics in summaries:
        aggregate["slo_attainment"].append(metrics.slo_attainment)
        aggregate["accuracy"].append(metrics.accuracy)
        aggregate["avg_response_time"].append(metrics.avg_response_time)
        aggregate["avg_scheduler_latency"].append(metrics.avg_scheduler_latency)
        aggregate["document_number"].append(metrics.document_number)
        aggregate["predict_count"].append(metrics.predict_count)
        aggregate["predict_samples"].append(metrics.predict_samples)
        aggregate["avg_confidence"].append(metrics.avg_confidence)

        lines.append(
            f"{experiment_name:30} | "
            f"{metrics.slo_attainment:6.3f} | "
            f"{metrics.accuracy:6.3f} | "
            f"{metrics.avg_response_time:9.3f} | "
            f"{metrics.avg_scheduler_latency:10.4f} | "
            f"{metrics.document_number:6d} | "
            f"{metrics.predict_count:8d} | "
            f"{metrics.predict_samples:9d} | "
            f"{metrics.avg_confidence:6.3f}"
        )

    lines.append("")
    lines.append("=== AVERAGES ACROSS EXPERIMENTS ===")
    for key in numeric_keys:
        values = aggregate[key]
        if not values:
            continue
        lines.append(f"{key}: {average(values):.4f}")

    batch_summary_path = batch_dir / "final_summary.txt"
    batch_summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {batch_summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun confidence plots and summaries for saved batch job outputs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/u/tchang85/dllm"),
        help="Repository root",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/u/tchang85/dllm/experiment_dir/20260420"),
        help="Root directory containing per-experiment run folders",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("/u/tchang85/dllm/sbatch_experiment_dir/20260420"),
        help="Root directory containing batch job folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be rerun without executing anything",
    )
    args = parser.parse_args()

    batch_dirs = [path for path in sorted(args.batch_root.iterdir()) if path.is_dir()]
    if not batch_dirs:
        raise SystemExit(f"No batch directories found under {args.batch_root}")

    for batch_dir in batch_dirs:
        config_path = batch_dir / "experiments_config.json"
        if not config_path.is_file():
            print(f"Skipping {batch_dir}: missing experiments_config.json")
            continue

        config = load_json(config_path)
        common = config.get("common", {})
        experiments = config.get("experiments", [])
        slo = float(common.get("SLO", 10))

        print(f"=== Batch {batch_dir.name} ===")
        experiment_rows: list[tuple[str, Path]] = []

        for index, experiment in enumerate(experiments):
            experiment_name = experiment.get("name") or experiment.get("experiment_name") or f"experiment_{index + 1}"
            run_dir = find_latest_run_dir(args.run_root, index, experiment_name)
            experiment_rows.append((experiment_name, run_dir))
            print(f"  {experiment_name} -> {run_dir}")

            if args.dry_run:
                continue

            rerun_run_dir(run_dir, args.repo_root)
            rerun_experiment_summary(run_dir, args.repo_root, slo)

        if not args.dry_run:
            rebuild_batch_final_summary(batch_dir, experiment_rows)


if __name__ == "__main__":
    main()