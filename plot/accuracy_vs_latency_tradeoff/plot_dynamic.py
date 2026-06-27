import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7.0,
    "lines.linewidth": 1.3,
})

# ----------------------------
# Data
# ----------------------------
portion = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
accuracy_raw = np.array([717, 726, 730, 732, 737, 743, 757, 759, 763, 777, 786])
steps = np.array([40.16, 42.14, 44.27, 46.6, 49.285, 52.18, 56.04, 59.84, 64.92, 71.15, 78.63])

accuracy = accuracy_raw / 10.0  # 717 → 71.7%

fig, ax = plt.subplots(figsize=(1.82, 1.34))

# Trend line
ax.plot(steps, accuracy, "--", color="gray", linewidth=1)

# Scatter
sc = ax.scatter(
    steps,
    accuracy,
    c=1 - portion,
    cmap="viridis",
    s=30,
    edgecolor="black",
    zorder=3,
)

# ----------------------------
# Axes settings
# ----------------------------
ax.set_xlabel("Number of denoising steps", labelpad=1)
ax.set_ylabel("Accuracy (%)", labelpad=1)
ax.tick_params(axis="both", direction="in", pad=1.0, length=1.5)
ax.grid(alpha=0.3)
ax.set_ylim(70, 80)
ax.set_xlim(38, 82)
ax.set_yticks([70, 72, 74, 76, 78, 80])
ax.set_xticks([40, 50, 60, 70, 80])

# Manually pack the axes to reduce dead border space on the small figure.
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.78)

# ----------------------------
# Horizontal colorbar ABOVE the plot, with a compact wrapped title
# ----------------------------
cbar_ax = fig.add_axes([0.20, 0.88, 0.72, 0.028])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation="horizontal")
cbar.ax.tick_params(labelsize=6.5, pad=1, length=1.5)
# set tick at 0, 0.2, 0.4, 0.6, 0.8, 1
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
# ax.tick_params(axis="both", direction="in", pad=2.0, length=1.5)

cbar_ax.set_title(
    "Fraction of steps using\nconfidence threshold = 0.6",
    fontsize=7.2,
    pad=1,
)

for spine in cbar_ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

plt.savefig("accuracy_vs_steps_dynamic.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig("accuracy_vs_steps_dynamic.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
