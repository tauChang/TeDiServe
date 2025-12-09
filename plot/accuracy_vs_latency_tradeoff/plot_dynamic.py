import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 2.5,
})

# ----------------------------
# Data
# ----------------------------
portion = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
accuracy_raw = np.array([717, 726, 730, 732, 737, 743, 757, 759, 763, 777, 786])
steps = np.array([40.16, 42.14, 44.27, 46.6, 49.285, 52.18, 56.04, 59.84, 64.92, 71.15, 78.63])

accuracy = accuracy_raw / 10.0  # 717 → 71.7%

fig, ax = plt.subplots(figsize=(7, 6))

# Trend line
ax.plot(steps, accuracy, "--", color="gray", linewidth=1)

# Scatter
sc = ax.scatter(
    steps,
    accuracy,
    c=1-portion,
    cmap="viridis",
    s=180,
    edgecolor="black",
    zorder=3,
)

# ----------------------------
# Axes settings
# ----------------------------
ax.set_xlabel("Number of denoising steps")
ax.set_ylabel("Accuracy (%)")
ax.grid(alpha=0.3)
ax.set_ylim(70, 80)
ax.set_xlim(38, 82)

# Leave room at the top for the colorbar
plt.tight_layout(rect=[0, 0, 1, 0.82])

# ----------------------------
# Horizontal colorbar ABOVE the plot, with label on top & frame
# ----------------------------
# [left, bottom, width, height] in figure coordinates
cbar_ax = fig.add_axes([0.15, 0.86, 0.7, 0.02])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")

# Put text ABOVE the bar
cbar_ax.set_title(
    "Fraction of steps using conf threshold = 0.6",
    fontsize=22,
    pad=10,
)

# Draw a frame around the colorbar
for spine in cbar_ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)

plt.savefig("accuracy_vs_steps_dynamic.png", dpi=300, bbox_inches="tight")
plt.savefig("accuracy_vs_steps_dynamic.pdf", dpi=300, bbox_inches="tight")
