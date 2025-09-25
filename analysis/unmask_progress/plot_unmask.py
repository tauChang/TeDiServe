import pandas as pd
import matplotlib.pyplot as plt

# Read the JSON-lines file
df = pd.read_json("GSAI-ML_LLaDA-8B-Base_prefix_step_estimator.json", lines=True)

# Compute % tokens unmasked
df["pct_unmasked"] = df["num_unmasked_tokens"] / df["output_length"] * 100

# Plot
plt.figure(figsize=(6,4))
plt.plot(df["num_denoise_ran"], df["pct_unmasked"], marker="o")
plt.xlabel("num_denoise_ran")
plt.ylabel("% tokens unmasked")
plt.title("Progress of Unmasking")
plt.grid(True)

plt.savefig("unmasking_progress.png")