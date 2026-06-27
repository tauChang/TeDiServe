import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


def parse_args():
    default_experiment_dirs = [
        Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260503/193754_sharegpt_20_infaas_only/0_infaas_20qps/"
        ),
        Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260503/163209_sharegpt_20_llumnix_only/0_llumnix_20qps/"
        ),
        # Path(
        #     "/u/tchang85/dllm/sbatch_experiment_dir/20260506/072221_sharegpt_20_tedi_no_batch_only/0_tedi_20qps"
        # ),
        Path(
            "/u/tchang85/dllm/sbatch_experiment_dir/20260506/114447_sharegpt_20_tedi_no_batch_only/0_tedi_20qps/"
        )
    ]
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary statistics for LLM judge scores, including a "
            "non-dropped-only subset based on benchmark latency info."
        )
    )
    parser.add_argument(
        "experiment_dirs",
        type=Path,
        nargs="*",
        default=default_experiment_dirs,
        help=(
            "One or more experiment directories containing "
            "logs/benchmark_latency_info.json and llm_judge_results.json "
            f"(default: {default_experiment_dirs})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the computed stats as JSON",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as file:
        return json.load(file)


def extract_request_statuses(benchmark_data):
    statuses = {}

    if isinstance(benchmark_data, list):
        if len(benchmark_data) != 1 or not isinstance(benchmark_data[0], dict):
            raise ValueError(
                "benchmark_latency_info.json list format must contain one object entry"
            )
        benchmark_data = benchmark_data[0]

    if not isinstance(benchmark_data, dict):
        raise ValueError(
            "benchmark_latency_info.json must contain a top-level JSON object or a one-element list"
        )

    request_records = benchmark_data.get("instances", benchmark_data)
    if not isinstance(request_records, dict):
        raise ValueError("benchmark_latency_info.json does not contain a valid instances map")

    for request_id, payload in request_records.items():
        if not isinstance(payload, dict):
            continue

        status = payload.get("status")
        if status is None:
            continue

        statuses[str(request_id)] = status

    return statuses


def extract_scores(judge_results):
    if not isinstance(judge_results, list):
        raise ValueError("llm_judge_results.json must contain a top-level JSON array")

    parsed_results = []
    for entry in judge_results:
        if not isinstance(entry, dict):
            continue

        request_id = entry.get("id")
        score = entry.get("score")
        if request_id is None or not isinstance(score, (int, float)):
            continue

        parsed_results.append(
            {
                "id": str(request_id),
                "score": float(score),
            }
        )

    return parsed_results


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def summarize_scores(scores):
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }

    ordered_scores = sorted(scores)
    return {
        "count": len(ordered_scores),
        "mean": mean(ordered_scores),
        "median": median(ordered_scores),
        "stddev": pstdev(ordered_scores),
        "min": ordered_scores[0],
        "max": ordered_scores[-1],
        "p25": percentile(ordered_scores, 0.25),
        "p75": percentile(ordered_scores, 0.75),
    }


def build_report(request_statuses, judge_scores):
    non_dropped_scores = []
    missing_in_benchmark = []
    dropped_count = 0

    for entry in judge_scores:
        request_id = entry["id"]
        score = entry["score"]

        status = request_statuses.get(request_id)
        if status is None:
            missing_in_benchmark.append(request_id)
            continue

        if status == "dropped":
            dropped_count += 1
        else:
            non_dropped_scores.append(score)

    return {
        "judge_result_count": len(judge_scores),
        "benchmark_request_count": len(request_statuses),
        "matched_request_count": len(judge_scores) - len(missing_in_benchmark),
        "missing_in_benchmark_count": len(missing_in_benchmark),
        "missing_in_benchmark_ids": missing_in_benchmark,
        "dropped_excluded_count": dropped_count,
        "non_dropped_score_stats": summarize_scores(non_dropped_scores),
    }


def analyze_experiment_dir(experiment_dir: Path):
    benchmark_latency_info = experiment_dir / "logs" / "benchmark_latency_info.json"
    judge_results = experiment_dir / "llm_judge_results.json"

    benchmark_data = load_json(benchmark_latency_info)
    judge_data = load_json(judge_results)

    request_statuses = extract_request_statuses(benchmark_data)
    judge_scores = extract_scores(judge_data)
    report = build_report(request_statuses, judge_scores)
    report["experiment_dir"] = str(experiment_dir)
    report["benchmark_latency_info"] = str(benchmark_latency_info)
    report["judge_results"] = str(judge_results)
    return report


def main():
    args = parse_args()

    reports = [analyze_experiment_dir(experiment_dir) for experiment_dir in args.experiment_dirs]

    output_payload = reports[0] if len(reports) == 1 else reports
    output_text = json.dumps(output_payload, indent=2, sort_keys=True)
    print(output_text)

    if args.output is not None:
        args.output.write_text(output_text + "\n")


if __name__ == "__main__":
    main()