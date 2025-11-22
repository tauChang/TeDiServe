import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define your file patterns
# files = [
#     # "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.5.json",
#     # "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.6.json",
#     # "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.7.json",
#     # "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.8.json",
#     # "GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9.json",
# ]
# dir = "../../step_data_kiet/gsm8k_100/256/"
# files = [dir + f for f in files]
files = [
    "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.9_256.json",
    "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.8_256.json",
    "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.7_256.json",
    "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.6_256.json",
    "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.5_256.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.9_512.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.8_512.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.7_512.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.6_512.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.5_512.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.9_1024.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.8_1024.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.7_1024.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.6_1024.json",
    # "/u/tchang85/dllm/step_data_dual_cache_with_output/dual_0.5_1024.json",

]

data_by_conf = defaultdict(list)

# Read each file and extract num_cur_unmasked_tokens
for path in files:
    # conf = float(path.split("conf")[-1].replace(".json", ""))
    values = []
    with open(path, "r") as f:
        # find conf by looking at the first row of the file in the "confidence_threshold" field
        first_line = f.readline()
        first_obj = json.loads(first_line)
        conf = first_obj["confidence_threshold"]
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            values.append(obj["num_cur_unmasked_tokens"])
    data_by_conf[conf].extend(values)

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