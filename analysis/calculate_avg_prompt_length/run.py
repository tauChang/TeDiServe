import json

def compute_avg_prompt_length(path):
    total = 0
    count = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # skip malformed lines
                continue

            req_id = obj.get("request_id", "")
            if req_id.startswith("cmpl-warmup"):
                # skip warmup requests entirely
                continue

            if "prompt_length" in obj:
                total += obj["prompt_length"]
                count += 1

    if count == 0:
        return 0.0

    print(count)
    return total / count


# Example usage:
# Example
# GSM8K
# file_path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251124/100951/workload_history.json"
# MMLU pro
# file_path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251124/210259/workload_history.json"
# MBPP
file_path = "/work2/10446/tchang85/stampede3/dllm/experiment_dir/20251126/144251/workload_history.json"
avg_len = compute_avg_prompt_length(file_path)
print("Average prompt length:", avg_len)
