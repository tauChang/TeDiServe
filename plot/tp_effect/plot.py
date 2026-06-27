import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.8,
    "lines.linewidth": 1.3,
})

# ---------------------------------------------
# Raw latency data (sec)
# ---------------------------------------------
# no_cache = [15.8626, 12.885, 8.21]
# cache_32 = [4.0237, 3.7519, 3.7031]
# cache_8  = [4.9792, 4.6268, 4.0981]
no_cache = [21.2932, 15.8892, 9.6108]
cache_32 = [3.0115, 2.5741, 2.1062]
cache_8  = [4.5685, 3.8012, 2.637]

# Normalize by TP=1 for each strategy
no_cache_norm = np.array(no_cache) / no_cache[0]
cache32_norm  = np.array(cache_32) / cache_32[0]
cache8_norm   = np.array(cache_8)  / cache_8[0]

print("No Cache (normalized):", no_cache_norm)
print("Cache (32) (normalized):", cache32_norm)
print("Cache (8) (normalized):", cache8_norm)

# Grouped data: values per TP across strategies
tp1 = [no_cache_norm[0], cache32_norm[0], cache8_norm[0]]
tp2 = [no_cache_norm[1], cache32_norm[1], cache8_norm[1]]
tp4 = [no_cache_norm[2], cache32_norm[2], cache8_norm[2]]

# ---------------------------------------------
# Plotting
# ---------------------------------------------
strategies = ["No Cache", "Cache (32)", "Cache (8)"]
x = np.arange(len(strategies))
width = 0.25

fig, ax = plt.subplots(figsize=(1.84, 1.42))

colors = ["#c7d9ff", "#7aa2ff", "#2c5bff"]

ax.bar(x - width, tp1, width, label="TP1", color=colors[0])
ax.bar(x,         tp2, width, label="TP2", color=colors[1])
ax.bar(x + width, tp4, width, label="TP4", color=colors[2])

ax.set_ylabel("Normalized latency", labelpad=1)
ax.set_xticks(x, strategies, fontsize=6.8)
ax.tick_params(axis="both", direction="in", pad=1.0, length=1.5)
ax.tick_params(axis="x", pad=2, length=2.5)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.legend(
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.57, 0.97),
    frameon=False,
    columnspacing=0.8,
    handletextpad=0.3,
)

fig.tight_layout(rect=[0, 0, 1, 0.83], pad=0.05)
plt.savefig("tp_effect_full_cg.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
plt.savefig("tp_effect_full_cg.pdf", bbox_inches="tight", pad_inches=0.01)
