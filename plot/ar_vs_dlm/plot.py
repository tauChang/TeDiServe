import matplotlib.pyplot as plt

OUTPUT_PNG = "ar_vs_dlm.png"
OUTPUT_PDF = "ar_vs_dlm.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "lines.linewidth": 2.5,
})

# Data
models = [
    "LLaDA 8B (0.9)",
    "LLaDA 8B (0.8)",
    "LLaDA 8B (0.7)",
    "LLaMA3 8B",
]

# throughput = [156.59, 183.07, 227.2, 108.2]
throughput = [218.82, 259.31, 299.61, 147.70]
accuracy = [78.8, 78.0, 77.0, 78.0]

# Colors: TeDiServe red, others default
colors = ["#90FCF9", "#7AD8E5", "#7699D4", "grey"]

fig, ax = plt.subplots(figsize=(3.45, 1.3))

# Per-point text offsets (dx, dy)
# Same order as models / throughput / accuracy
text_offsets = [
    (-26, 0.6),  # LLaDA (0.9)
    (-26, 0.6),  # LLaDA (0.8)
    (-26, 0.6),  # LLaDA (0.7)
    (-24, -0.7), # LLaMA-3-8B
]

for (m, x, y, c), (dx, dy) in zip(zip(models, throughput, accuracy, colors), text_offsets):
    ax.scatter(x, y, color=c, s=95, edgecolors='k', linewidths=1.2)
    ax.text(
        x + dx, y + dy,
        m,
        va="center",
        fontsize=8
    )


# Axes limits
ax.set_xlim(120, 345)
ax.set_ylim(76.5, 80)
# y tick at 77 78 79
# ax.set_yticks(range(77, 81, 1))
# x tick every 25
# ax.set_xticks(range(120, 370, 50))
# set x ticks at 150, 200, 250, 300, 350
ax.set_xticks([150, 200, 250, 300])

# Labels
ax.set_xlabel("Throughput (token/s)", labelpad=1)
ax.set_ylabel("Accuracy (%)", labelpad=1)
ax.set_yticks([77.0, 78.0, 79.0, 80])

# Grid
ax.grid(True, alpha=0.3)
ax.tick_params(axis="both", direction="in", pad=1.5, length=2)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
