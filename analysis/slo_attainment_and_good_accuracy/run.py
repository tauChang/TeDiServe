#!/usr/bin/env python3
import json
import pandas as pd
import argparse


# ==============================================================================
# Loading Helpers
# ==============================================================================

def load_data(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_response_times(data: dict) -> pd.DataFrame:
    """Load response_times → DataFrame(doc_id, response_time)."""
    return pd.DataFrame([
        {"doc_id": int(k), "response_time": v}
        for k, v in data["response_times"].items()
    ])


# ==============================================================================
# Task-Specific Sample Processors
# ==============================================================================

def process_gsm8k(data: dict, samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples[samples["filter"] == "flexible-extract"].copy()
    samples["is_correct"] = samples["exact_match"].astype(bool)
    return samples
# def process_gsm8k(data: dict, samples: pd.DataFrame) -> pd.DataFrame:
#     samples = samples[samples["filter"] == "strict-match"].copy()
#     samples["is_correct"] = samples["exact_match"].astype(bool)
#     return samples


def process_mbpp(data: dict, samples: pd.DataFrame) -> pd.DataFrame:
    samples["is_correct"] = samples["pass_at_1"].astype(bool)
    return samples


def process_mmlu_pro(data: dict, _samples_unused: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten all MMLU-Pro subtasks into a single DataFrame with:
        doc_id, category, is_correct
    """
    rows = []
    subtasks = data["group_subtasks"]["mmlu_pro"]

    for sub in subtasks:
        samples = data["samples"][sub]
        task_alias = data["configs"][sub]["task_alias"]  # e.g., "biology"

        for entry in samples:
            rows.append({
                "doc_id": int(entry["doc_id"]),
                "category": task_alias,
                "is_correct": bool(entry["exact_match"]),
            })

    return pd.DataFrame(rows)


# Dispatcher table
TASK_PROCESSORS = {
    "gsm8k": process_gsm8k,
    "mbpp": process_mbpp,
    "mbpp_instruct": process_mbpp,
    "mmlu_pro": process_mmlu_pro,
}


# ==============================================================================
# Metrics
# ==============================================================================

def compute_basic_metrics(df: pd.DataFrame, slo: float):
    total = len(df)
    slo_met = df[df["response_time"] <= slo]
    slo_miss = df[df["response_time"] > slo]

    # print 99th percentile response time
    p99_response_time = df["response_time"].quantile(0.99)
    print(f"99th percentile response time: {p99_response_time:.3f}s")
    p90_response_time = df["response_time"].quantile(0.90)
    print(f"90th percentile response time: {p90_response_time:.3f}s")

    return {
        "total": total,
        "slo_attainment": len(slo_met) / total if total > 0 else 0.0,
        "overall_accuracy": df["is_correct"].mean(),
        "avg_response_time": df["response_time"].mean(),
        "≤slo": {
            "count": len(slo_met),
            "accuracy": slo_met["is_correct"].mean()
        },
        ">slo": {
            "count": len(slo_miss),
            "accuracy": slo_miss["is_correct"].mean()
        },
    }


def compute_mmlu_pro_category_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a table (DataFrame):
        category | accuracy | n
    """
    return (
        df.groupby("category")["is_correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
        .sort_values("accuracy", ascending=False)
    )


# ==============================================================================
# Printing Helpers
# ==============================================================================

def print_basic_metrics(path: str, slo: float, m: dict):
    print(f"\n======================")
    print(f"File: {path}")
    print(f"Total samples: {m['total']}")
    print(f"SLO attainment (≤{slo}s): {m['slo_attainment']:.3f}")
    print(f"Overall accuracy: {m['overall_accuracy']:.3f}")
    print(f"Avg response time: {m['avg_response_time']:.3f}s\n")

    print(f"≤{slo}s:")
    print(f"  Count: {m['≤slo']['count']}")
    print(f"  Accuracy: {m['≤slo']['accuracy']:.3f}\n")

    print(f">{slo}s:")
    print(f"  Count: {m['>slo']['count']}")
    print(f"  Accuracy: {m['>slo']['accuracy']:.3f}")
    print(f"======================\n")



def print_mmlu_pro_category_summary(df: pd.DataFrame):
    table = compute_mmlu_pro_category_accuracy(df)
    print("\nMMLU-Pro per-category accuracy:")
    print(table.to_string())
    print()


# ==============================================================================
# Main Pipeline
# ==============================================================================

def analyze_eval_results(path: str, slo: float):
    data = load_data(path)
    response_times = load_response_times(data)

    # Determine task_name
    # For mmlu_pro there is no top-level "mmlu_pro" in samples, so detect manually.
    if "mmlu_pro" in data.get("group_subtasks", {}):
        task_name = "mmlu_pro"
        samples = None  # mmlu_pro uses subtasks only
    else:
        # normal tasks (gsm8k, mbpp, etc.)
        task_name = next(iter(data["samples"].keys()))
        samples = pd.DataFrame(data["samples"][task_name])
        samples["doc_id"] = samples["doc_id"].astype(int)

    # Process using task-specific processor
    processor = TASK_PROCESSORS[task_name]
    df = processor(data, samples)
    num_real_samples = len(df)

    # Merge response times
    df = pd.merge(df, response_times, on="doc_id", how="right")

    # generalize this for any doc_id beyond num_real_samples
    for doc_id in df['doc_id']:
        if doc_id > num_real_samples:
            df.loc[df['doc_id'] == doc_id, 'is_correct'] = df.loc[df['doc_id'] == (doc_id % num_real_samples), 'is_correct'].values[0]
    
    # --- Middle 20%–80% by arrival order ---
    # df_sorted = df.sort_values("doc_id")   # or sort by "arrival_index"
    # N = len(df_sorted)

    # start_idx = int(0.20 * N)
    # end_idx = int(1* N)

    # mid_df = df_sorted.iloc[start_idx:end_idx]
    # df = mid_df

    # Compute and print basic metrics
    metrics = compute_basic_metrics(df, slo)

    # Per-category table for mmlu_pro
    if task_name == "mmlu_pro":
        print_mmlu_pro_category_summary(df)

    print_basic_metrics(path, slo, metrics)



# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vLLM evaluation JSON results.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--slo", type=float, default=3.0)
    args = parser.parse_args()

    analyze_eval_results(args.path, slo=args.slo)
