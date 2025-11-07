import json
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_confidence_threshold(filename: str):
    # Read file
    records = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj["id"]
            records[rid].append(obj)

    plt.figure()
    # Process each id and plot it
    for rid, objs in records.items():
        # Sort by num_denoise_ran
        objs.sort(key=lambda x: x["num_denoise_ran"])

        # Max for normalization
        max_denoise = max(o["num_denoise_ran"] for o in objs)
        if max_denoise == 0:
            continue

        # Time normalized
        times = [o["num_denoise_ran"] / max_denoise for o in objs]

        # Confidence threshold
        conf = [o["confidence_threshold"] for o in objs]

        plt.plot(times, conf, marker="o", label=rid)

    plt.title("Confidence Threshold Over Time (One Line Per Request)")
    plt.xlabel("Normalized Time (num_denoise_ran / max)")
    plt.ylabel("Confidence Threshold")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # ✅ One PNG only
    plt.savefig("confidence_threshold.png")
    plt.close()


# Example usage:
file = "/u/tchang85/dllm/step_data_1106/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_prefix_suffix_block8_conf0.9.json"
plot_confidence_threshold(file)
