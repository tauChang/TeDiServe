import pandas as pd
import matplotlib.pyplot as plt
import math

# Files to plot (without .txt extension for convenience)
# files = ["dual_cache", "prefix_cache", "no_cache"]   # add or remove as needed
files = ["gpu_model_runner"]

n = len(files)            # number of columns = number of files
fig, axes = plt.subplots(
    2, n, figsize=(4*n, 6),   # width scales with number of files
    sharex='col',             # share x within each column
    sharey='row'              # share y within each row
)

# If there is only 1 column, axes will be 1D arrays; make them 2D for consistency
if n == 1:
    axes = axes.reshape(2, 1)

for col, name in enumerate(files):
    # Load and preprocess data
    df = pd.read_csv(f"{name}_profile.txt", header=None,
                     names=["batch_size", "exec_time"])
    df["id"] = range(len(df))
    df["exec_time"] *= 1000  # convert to ms

    # --- Top row: Execution time ---
    ax_top = axes[0, col]
    ax_top.plot(df["id"], df["exec_time"], color="blue", marker="o")
    ax_top.set_title(name)
    if col == 0:
        ax_top.set_ylabel("Execution Time (ms)", color="blue")
    ax_top.tick_params(axis="y", labelcolor="blue")
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # --- Bottom row: Batch size ---
    ax_bottom = axes[1, col]
    ax_bottom.plot(df["id"], df["batch_size"], color="green", marker="s")
    if col == 0:
        ax_bottom.set_ylabel("Batch Size", color="green")
    ax_bottom.set_xlabel("Iteration")
    ax_bottom.tick_params(axis="y", labelcolor="green")
    ax_bottom.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Optional: Save figure
fig.savefig("cache_interference.png", dpi=300)
