import json
import matplotlib.pyplot as plt
import numpy as np

# Define your file patterns
files = [
    "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.5.json",
    "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
    "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
    "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
    "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
]
dir = "../../step_data_kiet/gsm8k_100/256/"
files = [dir + f for f in files]

data_by_conf = {}

# Read each file and extract num_cur_unmasked_tokens
for path in files:
    conf = float(path.split("conf")[-1].replace(".json", ""))
    values = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            values.append(obj["num_cur_unmasked_tokens"])
    data_by_conf[conf] = values

# Plot
plt.figure(figsize=(8, 5))
bins = np.arange(0, max(max(v) for v in data_by_conf.values()) + 1) - 0.5

for conf, values in sorted(data_by_conf.items()):
    plt.hist(values, bins=bins, alpha=0.5, label=f"conf={conf}", density=True)

plt.xlabel("num_cur_unmasked_tokens")
plt.ylabel("Probability Density")
plt.title("Distribution of num_cur_unmasked_tokens by confidence threshold")
plt.legend(title="Confidence")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("unmask_distribution_by_confidence.png")

# print stats
for conf, values in sorted(data_by_conf.items()):
    print(f"Confidence: {conf}")
    print(f"  Mean: {np.mean(values):.2f}")
    print(f"  Median: {np.median(values):.2f}")
    print(f"  Std Dev: {np.std(values):.2f}")
    print(f"  Min: {np.min(values)}")
    print(f"  Max: {np.max(values)}")