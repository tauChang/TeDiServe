import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Path to your log file
filename = "/u/tchang85/dllm/step_data_1008/gsm8k_100/256/GSAI-ML_LLaDA-8B-Instruct_block32_conf0.9_slo3.json"  # one JSON per line

# Step 1: read file and parse lines
conf_data = defaultdict(list)
with open(filename, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            req_id = entry.get("id")
            conf = entry.get("confidence_threshold")
            if req_id is not None and conf is not None:
                conf_data[req_id].append(conf)
        except json.JSONDecodeError:
            print("Skipping invalid JSON line:", line)

# Step 2: compute average confidence per ID
avg_conf = {rid: sum(vals)/len(vals) for rid, vals in conf_data.items()}
print(len(avg_conf), "unique request IDs found.")

# Step 3: plot histogram
plt.figure(figsize=(6,4))
plt.hist(avg_conf.values(), bins=100, range=(0.5, 1), edgecolor="black")
# y-axis 0 - 100
plt.ylim(0, 100)

plt.title("Average Confidence Threshold per Request ID")
plt.xlabel("Average Confidence Threshold")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("confidence_distribution_3_new.png")
