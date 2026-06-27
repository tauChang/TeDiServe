import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# ---------------------------------------------
# Raw latency data (sec)
# ---------------------------------------------
no_cache = [21.2932, 15.8892, 9.6108]

# Normalize by TP=1
vals = np.array(no_cache) / no_cache[0]

labels = ["TP=1", "TP=2", "TP=4"]
colors = ["#c7d9ff", "#7aa2ff", "#2c5bff"]

plt.figure(figsize=(7,5))

plt.bar(labels, vals, color=colors, width=0.6)

plt.ylabel("Normalized Latency")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("tp_effect_no_cache.png", bbox_inches="tight", dpi=300)
plt.savefig("tp_effect_no_cache.pdf", bbox_inches="tight")