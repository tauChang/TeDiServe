#!/usr/bin/env python3
import json
import pandas as pd
import argparse


# ==============================================================================
# Loading Helpers
# ==============================================================================

def load_instances(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # data may be a list if you're appending results
    if isinstance(data, list):
        data = data[-1]   # take last entry

    instances = data["instances"]

    df = pd.DataFrame.from_dict(instances, orient="index")

    if "status" not in df.columns:
        df["status"] = "success"

    # Optional: compute total tokens
    df["total_tokens"] = df["prompt_len"] + df["expected_response_len"]

    # print latency stats
    print("Latency stats:")
    print(df["request_latency"].describe())
    
    # print prompt length stats
    print("Prompt length stats:")
    print(df["prompt_len"].describe())
    
    # print expected response length stats
    print("Expected response length stats:")
    print(df["expected_response_len"].describe())

    return df

def analyze_eval_results(json_path, slo=3.0):
    df = load_instances(json_path)
    # print stats
    print(f"Total requests: {len(df)}")
    print(f"Average latency: {df['request_latency'].mean():.2f} sec")

    total_requests = len(df)
    dropped_requests = (df["status"] == "dropped").sum()
    requests_within_slo = (
        (df["status"] == "success") &
        (df["request_latency"] <= slo)
    ).sum()
    dropped_percentage = dropped_requests / total_requests * 100.0
    slo_attainment = requests_within_slo / total_requests * 100.0

    print(f"Dropped requests: {dropped_percentage:.2f}% ({dropped_requests}/{total_requests})")
    print(f"SLO Attainment (@ {slo} sec): {slo_attainment:.2f}% ({requests_within_slo}/{total_requests})")
    
# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vLLM evaluation JSON results.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--slo", type=float, default=3.0)
    args = parser.parse_args()

    analyze_eval_results(args.path, slo=args.slo)
