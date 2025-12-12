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

# Baseline data (with KV cache)
models = [
    "LLaDA (0.9)",
    "LLaDA (0.8)",
    "LLaDA (0.7)",
    "LLaMA-3-8B",
]

throughput = [156.59, 183.07, 227.2, 108.2]
accuracy   = [78.8,   78.0,   77.0,  78.0]

colors = ["#90FCF9", "#7AD8E5", "#7699D4", "grey"]

# NEW — No-cache data
models_nc = [
    "LLaDA (0.9, no-cache)",
    "LLaDA (0.8, no-cache)",
    "LLaDA (0.7, no-cache)",
]

throughput_nc = [70.1, 90.0, 112.74]
accuracy_nc   = [78.9, 78.6, 77]  # <- fill in the accuracy for 0.7 if you have it

colors_nc = ["#3F8EBF", "#2E6FA3", "#1C4E78"]

fig, ax = plt.subplots(figsize=(10, 6))

# Offsets for text labels (you can tune these)
text_offsets = [
    (-15, 0.4),
    (-15, 0.4),
    (-22, 0.4),
    (-5, 0.4),
]

# Plot cached models
for (m, x, y, c), (dx, dy) in zip(zip(models, throughput, accuracy, colors), text_offsets):
    ax.scatter(x, y, color=c, s=270, edgecolors='k', linewidths=2.5)
    ax.text(x + dx, y + dy, m, va="center", fontsize=20)

# Plot NO-CACHE models (square markers)
for m, x, y, c in zip(models_nc, throughput_nc, accuracy_nc, colors_nc):
    if y is None:
        continue  # skip missing accuracy
    ax.scatter(
        x, y,
        color=c,
        s=270,
        marker="s",
        edgecolors='k',
        linewidths=2.5
    )
    ax.text(
        x - 20, y + 0.4,
        m,
        va="center",
        fontsize=20
    )

# Axes limits
ax.set_xlim(50, 250)
ax.set_ylim(76.5, 80)

# Ticks
ax.set_xticks(range(50, 251, 50))

# Labels
ax.set_xlabel("Throughput (token/s)")
ax.set_ylabel("Accuracy (%)")

# Grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ar_vs_dlm_no_cache.png", bbox_inches="tight")
plt.savefig("ar_vs_dlm_no_cache.pdf", bbox_inches="tight")
plt.show()
