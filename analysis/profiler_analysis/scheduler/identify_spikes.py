import json
import argparse

def find_large_totals(path, threshold=100.0):
    results = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total = entry.get("sections", {}).get("total")
            if total is not None and total > threshold:
                results.append(entry)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze scheduler overhead from profiler JSONL.")
    parser.add_argument("input", help="Path to profiler JSONL file")
    args = parser.parse_args()
    path = args.input
    entries = find_large_totals(path, threshold=100)

    print(f"Found {len(entries)} entries with total > 100 ms:")
    for e in entries:
        print(json.dumps(e, indent=2))
