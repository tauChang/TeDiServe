#!/usr/bin/env python3
import argparse
import json

def main(path):
    # Track maximum recompute count per request across all records
    max_recompute = {}

    # Read JSON-lines file
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            entry = json.loads(line)

            # request_num_recompute is a dict: {req_id: num}
            for req_id, num in entry.get("request_num_recompute", {}).items():
                # Ignore warmup requests
                if "warmup" in req_id:
                    continue

                # Keep the maximum seen across all lines
                if req_id not in max_recompute:
                    max_recompute[req_id] = num
                else:
                    max_recompute[req_id] = max(max_recompute[req_id], num)

    if not max_recompute:
        print("No non-warmup requests found.")
        return

    # Compute average
    avg_recompute = sum(max_recompute.values()) / len(max_recompute)
    min_recompute = min(max_recompute.values())
    max_recompute_value = max(max_recompute.values())

    print(f"Number of non-warmup requests: {len(max_recompute)}")
    print(f"Min recompute count: {min_recompute}")
    print(f"Max recompute count: {max_recompute_value}")
    print(f"Average max recompute count: {avg_recompute:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze recompute statistics.")
    parser.add_argument("--path", type=str, required=True, help="Path to JSONL log file")
    args = parser.parse_args()
    main(args.path)
