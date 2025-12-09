import matplotlib.pyplot as plt

# -----------------------------
# Data
# -----------------------------
rps = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

tp4x2 = [1.937, 1.950, 1.998, 2.636, 5.371, 11.371]
tp1x8 = [3.367, 3.367, 3.384, 3.385, 3.386, 3.390]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22,
    "lines.linewidth": 2.5,
})


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7, 5))

plt.plot(rps, tp4x2, marker='o', label="TP=4×2", markersize=10, color='#5D688A')
plt.plot(rps, tp1x8, marker='o', label="TP=1×8", markersize=10, color='#F7A5A5')

plt.xlabel("Requests per Second")
plt.ylabel("Latency (sec)")
plt.ylim(0, 12)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig("config_comparison.png", bbox_inches='tight', dpi=300)
plt.savefig("config_comparison.pdf", bbox_inches='tight')