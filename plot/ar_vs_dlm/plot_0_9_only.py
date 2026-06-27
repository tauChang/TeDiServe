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

# Data
models = [
    "LLaDA 8B",
    # "LLaDA 8B (0.9)",
    # "LLaDA 8B (0.8)",
    # "LLaDA 8B (0.7)",
    "LLaMA3 8B",
]

throughput = [
    # 156.59, 
    183.07, 
    # 227.2, 
    108.2
]
accuracy = [
    # 78.8, 
    78.0, 
    # 77.0, 
    78.0]

# Colors: TeDiServe red, others default
colors = [
    # "#90FCF9", 
    "#7AD8E5", 
    # "#7699D4", 
    "grey"]

fig, ax = plt.subplots(figsize=(8, 3.5))

# Per-point text offsets (dx, dy)
# Same order as models / throughput / accuracy
text_offsets = [
    (-17, 0.5),   # LLaDA (0.9)
    # (-17, 0.5),  # LLaDA (0.8)
    # (-31, 0.5),  # LLaDA (0.7)
    (-5, -0.5),   # LLaMA-3-8B
]

for (m, x, y, c), (dx, dy) in zip(zip(models, throughput, accuracy, colors), text_offsets):
    ax.scatter(x, y, color=c, s=270, edgecolors='k', linewidths=2.5)
    ax.text(
        x + dx, y + dy,
        m,
        va="center",
        fontsize=20
    )


# Axes limits
ax.set_xlim(100, 250)
ax.set_ylim(76.5, 80)
# y tick at 77 78 79
# ax.set_yticks(range(77, 81, 1))
# x tick every 25
ax.set_xticks(range(100, 251, 50))

# Labels
ax.set_xlabel("Throughput (token/s)")
ax.set_ylabel("GSM8K\nAccuracy (%)")
ax.set_yticks([77.0, 78.0, 79.0, 80])

# Grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ar_vs_dlm_0_9_only.png", bbox_inches="tight")
plt.savefig("ar_vs_dlm_0_9_only.pdf", bbox_inches="tight")
plt.show()
