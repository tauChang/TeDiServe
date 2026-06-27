import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
rps = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

tp4x2 = [1.937, 1.950, 1.998, 2.636, 5.371, 11.371]
tp1x8 = [3.367, 3.367, 3.384, 3.385, 3.386, 3.390]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7    ,
    "legend.fontsize": 7,
    "lines.linewidth": 1.4,
})


# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(1.84, 1.42))

ax.plot(rps, tp4x2, marker='o', label="TP4×2", markersize=4.4, color='#5D688A')
ax.plot(rps, tp1x8, marker='o', label="TP1×8", markersize=4.4, color='#F7A5A5')

ax.set_xlabel("Requests per second", labelpad=1)
ax.set_ylabel("Latency (sec)", labelpad=1)
ax.tick_params(axis="both", direction="in", pad=2.0, length=1.5)
ax.set_ylim(0, 12)
ax.set_xlim(0.18, 1.57)
ax.set_xticks(rps)
ax.set_yticks([0, 2, 4, 6, 8, 10, 12])

ax.grid(True, linestyle="--", alpha=0.5)
fig.legend(
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.978),
    frameon=False,
    columnspacing=0.8,
    handlelength=1.5,
    handletextpad=0.35,
)

fig.tight_layout(rect=[0, 0, 1, 0.86], pad=0.15)
plt.savefig("config_comparison.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.savefig("config_comparison.pdf", bbox_inches='tight', pad_inches=0.01)