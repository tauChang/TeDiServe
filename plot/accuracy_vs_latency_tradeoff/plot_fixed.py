import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7.0,
    "lines.linewidth": 1.3,
})

# --------------------------------------
# Data
# --------------------------------------
conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
accuracy_raw = np.array([786, 789, 770, 717, 659])
steps = np.array([78.63, 60.92, 48.72, 40.16, 34.92])

accuracy = accuracy_raw / 10.0   # 786 → 78.6%

colors = {
    0.9: "#303D2B",
    0.8: "#5B7553",
    0.7: "#8EB897",
    0.6: "#A9D0AA",
    0.5: "#E3F4E1",
}

# --------------------------------------
# Plot
# --------------------------------------
fig, ax = plt.subplots(figsize=(1.82, 1.42))

for c, acc, st in zip(conf, accuracy, steps):
    ax.scatter(
        st,
        acc,
        color=colors[c],
        edgecolor="black",
        s=52,
        zorder=3
    )

# Trend line
order = np.argsort(steps)
ax.plot(steps[order], accuracy[order], "--", color="gray", linewidth=1)

legend_order = [0.5, 0.6, 0.7, 0.8, 0.9]
legend_handles = [
    Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markerfacecolor=colors[c],
        markeredgecolor="black",
        markersize=5.0,
        label=f"{c:.1f}",
    )
    for c in legend_order
]
legend = fig.legend(
    legend_handles,
    [handle.get_label() for handle in legend_handles],
    title="Confidence threshold",
    loc="upper center",
    bbox_to_anchor=(0.565, 1.02),
    ncol=5,
    frameon=True,
    handlelength=0.6,
    handletextpad=0.28,
    labelspacing=0.1,
    columnspacing=0.72,
    borderpad=0.35,
)
legend.get_title().set_ha("center")
legend.get_title().set_fontsize(6.8)

ax.set_xlabel("Number of denoising steps", labelpad=1)
ax.set_ylabel("Accuracy (%)", labelpad=1)
ax.tick_params(axis="both", direction="in", pad=2.0, length=1.5)
ax.grid(alpha=0.3)
ax.set_ylim(64, 80)
ax.set_xlim(28, 82)
ax.set_yticks([65, 70, 75, 80])
ax.set_xticks([30, 40, 50, 60, 70, 80])

fig.tight_layout(rect=[0, 0, 1, 0.81], pad=0.02)
plt.savefig("accuracy_vs_steps_fixed.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
plt.savefig("accuracy_vs_steps_fixed.pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
