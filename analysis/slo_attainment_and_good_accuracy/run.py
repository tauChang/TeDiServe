import json
import pandas as pd

# --- Load JSON ---------------------------------------------------------------
path = "/u/tchang85/dllm/eval/results_1024_default_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results_1024_no_opp_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results_1024_cs_3/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results_1024_default_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results_1024_no_opp_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"
# path = "/u/tchang85/dllm/eval/results_1024_cs_1.5/gsm8k_/256/GSAI-ML_LLaDA-8B-Instruct_block32.json"

with open(path, "r") as f:
    data = json.load(f)

# --- Flatten response times --------------------------------------------------
response_times = pd.DataFrame([
    {"doc_id": int(k), "response_time": v}
    for k, v in data["response_times"].items()
])

# --- Flatten samples ---------------------------------------------------------
samples = pd.DataFrame(data["samples"]["gsm8k"])
# Keep only flexible-extract samples
samples = samples[samples["filter"] == "flexible-extract"].copy()
samples["doc_id"] = samples["doc_id"].astype(int)

# --- Merge and compute fields ------------------------------------------------
df = pd.merge(samples, response_times, on="doc_id", how="left")

# Compute accuracy column
df["is_correct"] = df["exact_match"].astype(bool)

# --- Print and example queries ----------------------------------------------
SLO = 5
print(f"Total flexible-extract samples: {len(df)}")
print(f"SLO attainment (response times ≤{SLO}s): {len(df[df['response_time'] <= SLO]) / len(df):.3f}")
print(f"Overall accuracy: {df['is_correct'].mean():.3f}")
print(f"Average response time (s): {df['response_time'].mean():.3f}")
print(f"Number of samples with response times ≤{SLO}s: {len(df[df['response_time'] <= SLO])}")
print(f"Accuracy for response times ≤5s: {df[df['response_time'] <= SLO]['is_correct'].mean():.3f}")
print(f"Number of samples with response times >{SLO}s: {len(df[df['response_time'] > SLO])}")
print(f"Accuracy for response times >5s: {df[df['response_time'] > SLO]['is_correct'].mean():.3f}")
# print("\nAccuracy by response-time bin (seconds):")
# print(summary)
