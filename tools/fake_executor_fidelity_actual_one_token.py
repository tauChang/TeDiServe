#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


GPU_COUNTS = (1, 2, 4, 8, 16)
SBATCH_JOB_RE = re.compile(r"Submitted batch job (?P<job_id>\d+)")
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
class LaunchRow:
    gpu_count: int
    request_per_gpu: int
    total_requests: int
    num_fewshot: int | str
    thresholds: str
    config_path: Path
    job_id: str | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    slo_attainment: float
    accuracy: float
    avg_response_time: float
    avg_scheduler_latency: float
    document_number: int
    predict_count: int
    predict_samples: int
    avg_confidence: float


@dataclass(frozen=True)
class SummaryRow:
    gpu_count: int
    request_per_gpu: int
    total_requests: int
    num_fewshot: int | str
    thresholds: str
    batch_dir: Path
    metrics: SummaryMetrics


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path_for_gpu(gpu_count: int) -> Path:
    return repo_root() / "sbatch_config" / f"fake_executor_fidelity_actual_one_token_{gpu_count}.json"


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def normalize_thresholds(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        normalized.extend(parts)
    if not normalized:
        raise ValueError("candidate confidence thresholds cannot be empty")
    return normalized


def parse_total_requests(arrival_pattern: str) -> int:
    total_requests = 0
    for segment in arrival_pattern.split(","):
        segment = segment.strip()
        if not segment:
            continue
        parts = segment.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid arrival pattern segment '{segment}'. Expected num:rps or num:rps:cv"
            )
        total_requests += int(parts[0].strip())
    return total_requests


def parse_experiment_summary_log(path: Path) -> SummaryMetrics:
    text = path.read_text()

    def extract(name: str) -> str:
        match = EXPERIMENT_SUMMARY_RE[name].search(text)
        if not match:
            raise ValueError(f"Missing {name} in {path}")
        return match.group(1)

    return SummaryMetrics(
        slo_attainment=float(extract("slo_attainment")),
        accuracy=float(extract("accuracy")),
        avg_response_time=float(extract("avg_response_time")),
        avg_scheduler_latency=float(extract("avg_scheduler_latency")),
        document_number=int(extract("document_number")),
        predict_count=int(extract("predict_count")),
        predict_samples=int(extract("predict_samples")),
        avg_confidence=float(extract("avg_confidence")),
    )


def parse_final_summary(path: Path) -> SummaryMetrics:
    for line in path.read_text().splitlines():
        match = SUMMARY_ROW_RE.match(line)
        if not match:
            continue
        return SummaryMetrics(
            slo_attainment=float(match.group("slo")),
            accuracy=float(match.group("acc")),
            avg_response_time=float(match.group("avg_rt")),
            avg_scheduler_latency=float(match.group("sched")),
            document_number=int(match.group("docs")),
            predict_count=int(match.group("pred_cnt")),
            predict_samples=int(match.group("pred_samp")),
            avg_confidence=float(match.group("conf")),
        )
    raise ValueError(f"Could not find a summary row in {path}")


def update_config(
    path: Path,
    gpu_count: int,
    request_per_gpu: int | None,
    num_fewshot: int | None,
    candidate_confidence_thresholds: Sequence[str] | None,
    dry_run: bool,
) -> LaunchRow:
    payload = load_json(path)
    common = payload.setdefault("common", {})

    if request_per_gpu is not None:
        total_requests = request_per_gpu * gpu_count
        common["ARRIVAL_PATTERN"] = f"{total_requests}:1000"
    else:
        total_requests = parse_total_requests(common["ARRIVAL_PATTERN"])
        request_per_gpu = total_requests // gpu_count

    common["NUM_CONCURRENT"] = total_requests

    if num_fewshot is not None:
        common["NUM_FEWSHOT"] = num_fewshot

    if candidate_confidence_thresholds is not None:
        thresholds = normalize_thresholds(candidate_confidence_thresholds)
        common["CANDIDATE_CONFIDENCE_THRESHOLDS"] = thresholds
        default_threshold = str(common.get("DEFAULT_CONFIDENCE_THRESHOLD"))
        if default_threshold not in thresholds:
            common["DEFAULT_CONFIDENCE_THRESHOLD"] = float(thresholds[0])

    if not dry_run:
        dump_json(path, payload)

    thresholds = ",".join(str(value) for value in common["CANDIDATE_CONFIDENCE_THRESHOLDS"])
    return LaunchRow(
        gpu_count=gpu_count,
        request_per_gpu=request_per_gpu,
        total_requests=total_requests,
        num_fewshot=common.get("NUM_FEWSHOT", "N/A"),
        thresholds=thresholds,
        config_path=path,
    )


def submit_config(path: Path) -> str | None:
    command = [
        sys.executable,
        str(repo_root() / "submit_job.py"),
        str(path),
        str(repo_root() / "run_lmeval_multi_sbatch.sh"),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    match = SBATCH_JOB_RE.search(result.stdout)
    return match.group("job_id") if match else None


def latest_batch_dir(base_dir: Path, config_stem: str, date: str | None) -> Path | None:
    if date is not None:
        day_dirs = [base_dir / date]
    else:
        day_dirs = [path for path in base_dir.iterdir() if path.is_dir()]

    matches: list[Path] = []
    suffix = f"_{config_stem}"
    for day_dir in day_dirs:
        if not day_dir.is_dir():
            continue
        for candidate in day_dir.iterdir():
            if candidate.is_dir() and candidate.name.endswith(suffix):
                matches.append(candidate)

    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def find_run_dir(batch_dir: Path, experiment_name: str) -> Path:
    matches = [path for path in batch_dir.iterdir() if path.is_dir() and path.name.endswith(f"_{experiment_name}")]
    if not matches:
        raise FileNotFoundError(f"Could not find run directory for {experiment_name} under {batch_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def collect_summary_row(batch_dir: Path) -> SummaryRow:
    config = load_json(batch_dir / "experiments_config.json")
    common = config["common"]
    experiment_name = config["experiments"][0]["name"]
    gpu_count = int(config["sbatch"]["gpus"])
    total_requests = parse_total_requests(common["ARRIVAL_PATTERN"])
    request_per_gpu = total_requests // gpu_count
    thresholds = ",".join(str(value) for value in common["CANDIDATE_CONFIDENCE_THRESHOLDS"])

    final_summary_path = batch_dir / "final_summary.txt"
    if final_summary_path.is_file():
        metrics = parse_final_summary(final_summary_path)
    else:
        run_dir = find_run_dir(batch_dir, experiment_name)
        metrics = parse_experiment_summary_log(run_dir / "logs" / "experiment_summary.log")

    return SummaryRow(
        gpu_count=gpu_count,
        request_per_gpu=request_per_gpu,
        total_requests=total_requests,
        num_fewshot=common.get("NUM_FEWSHOT", "N/A"),
        thresholds=thresholds,
        batch_dir=batch_dir,
        metrics=metrics,
    )


def format_launch_rows(rows: Sequence[LaunchRow]) -> str:
    lines = [
        "gpu | req/gpu | total | fewshot | thresholds | config",
        "-" * 120,
    ]
    for row in rows:
        lines.append(
            f"{row.gpu_count:>3} | "
            f"{row.request_per_gpu:>7} | "
            f"{row.total_requests:>5} | "
            f"{str(row.num_fewshot):>7} | "
            f"{row.thresholds:<23} | "
            f"{row.config_path.name}"
        )
    return "\n".join(lines)


def format_summary_rows(rows: Sequence[SummaryRow]) -> str:
    lines = [
        "gpu | req/gpu | total | fewshot | slo | acc | avg_rt(s) | sched(ms) | conf | thresholds | batch",
        "-" * 160,
    ]
    for row in rows:
        metrics = row.metrics
        lines.append(
            f"{row.gpu_count:>3} | "
            f"{row.request_per_gpu:>7} | "
            f"{row.total_requests:>5} | "
            f"{str(row.num_fewshot):>7} | "
            f"{metrics.slo_attainment:>5.3f} | "
            f"{metrics.accuracy:>5.3f} | "
            f"{metrics.avg_response_time:>9.3f} | "
            f"{metrics.avg_scheduler_latency:>9.4f} | "
            f"{metrics.avg_confidence:>4.3f} | "
            f"{row.thresholds:<23} | "
            f"{row.batch_dir.name}"
        )
    return "\n".join(lines)


def run_launch(args: argparse.Namespace) -> None:
    rows: list[LaunchRow] = []
    for gpu_count in GPU_COUNTS:
        row = update_config(
            config_path_for_gpu(gpu_count),
            gpu_count,
            args.request_per_gpu,
            args.num_fewshot,
            args.candidate_confidence_thresholds,
            args.dry_run,
        )
        rows.append(row)

    print(format_launch_rows(rows))

    if args.dry_run:
        return

    print("\nSubmitting jobs:")
    for row in rows:
        job_id = submit_config(row.config_path)
        if job_id is None:
            print(f"  {row.config_path.name}: submitted")
        else:
            print(f"  {row.config_path.name}: job {job_id}")


def run_summary(args: argparse.Namespace) -> None:
    batch_dirs: list[Path] = []
    if args.batch_dirs:
        batch_dirs = [Path(path).resolve() for path in args.batch_dirs]
    else:
        base_dir = Path(args.base_dir).resolve()
        for gpu_count in GPU_COUNTS:
            print(f"Finding batch for GPU count {gpu_count}...", end=" ", flush=True)
            config_stem = config_path_for_gpu(gpu_count).stem
            batch_dir = latest_batch_dir(base_dir, config_stem, args.date)
            if batch_dir is None:
                raise FileNotFoundError(
                    f"Could not find a batch directory for {config_stem} under {base_dir}"
                )
            batch_dirs.append(batch_dir)

    rows = [collect_summary_row(batch_dir) for batch_dir in batch_dirs]
    rows.sort(key=lambda row: row.gpu_count)

    output = format_summary_rows(rows)
    print(output)

    if args.output is not None:
        output_path = Path(args.output).resolve()
        output_path.write_text(output + "\n")
        print(f"\nWrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch and summarize fake executor fidelity one-token jobs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser(
        "launch",
        help="Update the fidelity configs and submit the five jobs.",
    )
    launch_parser.add_argument(
        "--request-per-gpu",
        type=int,
        default=None,
        help="Requests per GPU. Total requests become request_per_gpu * gpu_count.",
    )
    launch_parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="NUM_FEWSHOT to write into each config.",
    )
    launch_parser.add_argument(
        "--candidate-confidence-thresholds",
        nargs="+",
        default=None,
        help="Candidate confidence thresholds as space-separated values or a comma-separated string.",
    )
    launch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview config changes without writing files or submitting jobs.",
    )
    launch_parser.set_defaults(func=run_launch)

    summary_parser = subparsers.add_parser(
        "summary",
        help="Print one combined summary for the latest or specified fidelity batches.",
    )
    summary_parser.add_argument(
        "--base-dir",
        default=str(repo_root() / "sbatch_experiment_dir"),
        help="Base sbatch experiment directory to search for the latest matching batches.",
    )
    summary_parser.add_argument(
        "--date",
        default=None,
        help="Optional YYYYMMDD subdirectory under sbatch_experiment_dir to constrain the search.",
    )
    summary_parser.add_argument(
        "--batch-dirs",
        nargs="+",
        default=None,
        help="Explicit batch directories to summarize instead of auto-discovering the latest ones.",
    )
    summary_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the combined summary table.",
    )
    summary_parser.set_defaults(func=run_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()