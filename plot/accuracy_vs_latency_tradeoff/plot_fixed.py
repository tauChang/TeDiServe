import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 2.5,
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
fig, ax = plt.subplots(figsize=(7, 5))

for c, acc, st in zip(conf, accuracy, steps):
    ax.scatter(
        st,
        acc,
        color=colors[c],
        edgecolor="black",
        s=180,
        label=f"{c}",   # Just the numeric confidence value
        zorder=3
    )

# Trend line
order = np.argsort(steps)
ax.plot(steps[order], accuracy[order], "--", color="gray", linewidth=1)

# Legend (outside plot)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
# order from low to high confidence
unique = dict(sorted(unique.items(), key=lambda item: float(item[0])))
# legend = ax.legend(
#     unique.values(),
#     unique.keys(),
#     title="Confidence\nThreshold",
#     frameon=True,
#     loc="center left",
#     bbox_to_anchor=(1.02, 0.5)
# )
# legend._legend_title_box._children[0].set_ha("center")
legend = ax.legend(
    unique.values(),
    unique.keys(),
    title="Confidence Threshold",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.4),  # move legend further above figure
    ncol=len(unique),
    frameon=True,
    handletextpad=-0.2,
    columnspacing=0.5,
)
legend._legend_title_box._children[0].set_ha("center")

plt.tight_layout(rect=[0, 0, 1, 0.92])  # reserve extra space so legend never overlaps



ax.set_xlabel("Number of denoising steps")
ax.set_ylabel("Accuracy (%)")
ax.grid(alpha=0.3)
# set y limit to 65 to 80
ax.set_ylim(65, 80)
ax.set_xlim(30, 82)

plt.tight_layout()
plt.savefig("accuracy_vs_steps_fixed.png", dpi=300, bbox_inches="tight")
plt.savefig("accuracy_vs_steps_fixed.pdf", dpi=300, bbox_inches="tight")
