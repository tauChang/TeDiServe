#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def compute_eval_metrics(result_path: Path, slo: float) -> dict:
    data = load_json(result_path)
    response_times = data.get("response_times", {})
    total = len(response_times)
    avg_response_time = sum(response_times.values()) / total if total else 0.0
    slo_attainment = (
        sum(1 for value in response_times.values() if value <= slo) / total
        if total else 0.0
    )

    if "mmlu_pro" in data.get("group_subtasks", {}):
        rows = []
        for subtask in data["group_subtasks"]["mmlu_pro"]:
            task_alias = data["configs"][subtask]["task_alias"]
            for entry in data["samples"][subtask]:
                rows.append(
                    {
                        "doc_id": int(entry["doc_id"]),
                        "category": task_alias,
                        "is_correct": bool(entry["exact_match"]),
                    }
                )
        df = pd.DataFrame(rows)
    else:
        task_name = next(iter(data["samples"].keys()))
        df = pd.DataFrame(data["samples"][task_name])
        df["doc_id"] = df["doc_id"].astype(int)

        if task_name == "gsm8k":
            df = df[df["filter"] == "flexible-extract"].copy()
            df["is_correct"] = df["exact_match"].astype(bool)
        elif task_name in {"mbpp", "mbpp_instruct"}:
            df["is_correct"] = df["pass_at_1"].astype(bool)
        else:
            df["is_correct"] = df["exact_match"].astype(bool)

    if not df.empty:
        response_time_df = pd.DataFrame(
            [{"doc_id": int(k), "response_time": v} for k, v in response_times.items()]
        )
        df = pd.merge(df[["doc_id", "is_correct"]], response_time_df, on="doc_id", how="right")

        if len(df) > 0:
            num_docs = len(df)
            for doc_id in df["doc_id"]:
                if doc_id > num_docs:
                    df.loc[df["doc_id"] == doc_id, "is_correct"] = df.loc[
                        df["doc_id"] == (doc_id % num_docs), "is_correct"
                    ].values[0]

    accuracy = float(df["is_correct"].mean()) if not df.empty else 0.0

    dropped_requests = data.get("dropped_requests", [])
    dropped_percentage = (len(dropped_requests) / total * 100) if total > 0 else 0.0

    return {
        "document_number": total,
        "slo_attainment": slo_attainment,
        "accuracy": accuracy,
        "avg_response_time": avg_response_time,
        "dropped_percentage": dropped_percentage,
    }


def parse_scheduler_summary(path: Path):
    if not path.is_file():
        return None
    text = path.read_text()
    match = re.search(r"total_avg_latency\s*=\s*([0-9.]+)\s*ms", text)
    return float(match.group(1)) if match else None


def parse_predict_summary(path: Path) -> dict:
    if not path.is_file():
        return {}

    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("count"):
            values["predict_count"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("total_samples_processed"):
            values["predict_samples"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("avg(ms)"):
            values["predict_avg_ms"] = float(line.split(":", 1)[1].strip())

    return values


def parse_confidence_stats(path: Path):
    if not path.is_file():
        return None
    data = load_json(path)
    aggregate = data.get("aggregate", {})
    value = aggregate.get("overall_avg_confidence_threshold")
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact summary for one experiment run.")
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--slo", type=float, required=True)
    parser.add_argument("--scheduler-summary", required=False)
    parser.add_argument("--predict-summary", required=False)
    parser.add_argument("--confidence-stats", required=False)
    args = parser.parse_args()

    result_path = Path(args.result_path)
    metrics = compute_eval_metrics(result_path, args.slo)

    scheduler_latency = parse_scheduler_summary(Path(args.scheduler_summary)) if args.scheduler_summary else None
    predict_metrics = parse_predict_summary(Path(args.predict_summary)) if args.predict_summary else {}
    avg_confidence = parse_confidence_stats(Path(args.confidence_stats)) if args.confidence_stats else None

    print("=== EXPERIMENT SUMMARY ===")
    print(f"SLO attainment (≤{args.slo:g}s): {metrics['slo_attainment']:.3f}")
    print(f"Dropped percentage: {metrics['dropped_percentage']:.2f}%")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Average response time: {metrics['avg_response_time']:.3f}s")
    if scheduler_latency is not None:
        print(f"Average scheduler overhead / latency: {scheduler_latency:.4f} ms")
    else:
        print("Average scheduler overhead / latency: N/A")
    print(f"Document number: {metrics['document_number']}")
    print(f"Predict count: {predict_metrics.get('predict_count', 'N/A')}")
    print(f"Predict samples: {predict_metrics.get('predict_samples', 'N/A')}")
    if avg_confidence is not None:
        print(f"Average confidence: {avg_confidence:.4f}")
    else:
        print("Average confidence: N/A")


if __name__ == "__main__":
    main()