import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load JSONs --------------------------------------------------------------
paths = {
    # "Ours": "/u/tchang85/dllm/eval/results_1024_default_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
    # "Ours w/o opp.": "/u/tchang85/dllm/eval/results_1024_no_opp_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
    # "Naive": "/u/tchang85/dllm/eval/results_1024_cs_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
    "Ours": "/u/tchang85/dllm/eval/results_1024_default_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
    "Ours w/o opp.": "/u/tchang85/dllm/eval/results_1024_no_opp_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",
    "Naive": "/u/tchang85/dllm/eval/results_1024_cs_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json",   
}

# --- Collect response times --------------------------------------------------
results = {}
plt.rcParams.update({'font.size': 14})  # Increase font size globally

for name, path in paths.items():
    with open(path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame([
        {"doc_id": int(k), "response_time": v}
        for k, v in data["response_times"].items()
    ])
    results[name] = df["response_time"].values

# --- Plot CDFs ---------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.xlim(0, 16)  # Set x-axis range from 0 to 16

for name, times in results.items():
    sorted_t = np.sort(times)
    cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
    plt.plot(sorted_t, cdf, label=name, linewidth=2)

plt.axvline(x=5, color='red', linestyle='--', linewidth=1.5)

plt.xlabel("Response Time (s)")
plt.ylabel("CDF")
plt.title("Response Time Distribution (CDF)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("response_time_cdf.png")
