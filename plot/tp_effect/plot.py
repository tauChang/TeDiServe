import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 2.5,
})

# ---------------------------------------------
# Raw latency data (sec)
# ---------------------------------------------
no_cache = [15.8626, 12.885, 8.21]
cache_32 = [4.0237, 3.7519, 3.7031]
cache_8  = [4.9792, 4.6268, 4.0981]

# Normalize by TP=1 for each strategy
no_cache_norm = np.array(no_cache) / no_cache[0]
cache32_norm  = np.array(cache_32) / cache_32[0]
cache8_norm   = np.array(cache_8)  / cache_8[0]

# Grouped data: values per TP across strategies
tp1 = [no_cache_norm[0], cache32_norm[0], cache8_norm[0]]
tp2 = [no_cache_norm[1], cache32_norm[1], cache8_norm[1]]
tp4 = [no_cache_norm[2], cache32_norm[2], cache8_norm[2]]

# ---------------------------------------------
# Plotting
# ---------------------------------------------
strategies = ["No Cache", "Cache (32)", "Cache (8)"]
x = np.arange(len(strategies))
width = 0.2

plt.figure(figsize=(7, 5))

colors = ["#c7d9ff", "#7aa2ff", "#2c5bff"]

plt.bar(x - width, tp1, width, label="TP=1", color=colors[0])
plt.bar(x,         tp2, width, label="TP=2", color=colors[1])
plt.bar(x + width, tp4, width, label="TP=4", color=colors[2])

plt.ylabel("Normalized Latency")
plt.xticks(x, strategies)

plt.grid(axis="y", linestyle="--", alpha=0.4)

# --- Legend outside the plot ---------------
# plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.legend(
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.95),
    frameon=False,
    columnspacing=1,
    handletextpad=0.3,
)

plt.tight_layout()
plt.savefig("tp_effect.png", bbox_inches="tight", dpi=300)
plt.savefig("tp_effect.pdf", bbox_inches="tight")
